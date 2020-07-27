SRC=en
TGT=zh

N_THREADS=8

MAIN_PATH=$PWD

DATA_PATH=$MAIN_PATH/data
PROC_PATH=$DATA_PATH/processed

mkdir $PROC_PATH

# tools
TOOLS_PATH=$MAIN_PATH/tools
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast
MOSES=$TOOLS_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl

# BPE / vocab files
BPE_CODES=$PROC_PATH/precalc_vocabs/codes
SRC_VOCAB=$PROC_PATH/precalc_vocabs/vocab.$SRC
TGT_VOCAB=$PROC_PATH/precalc_vocabs/vocab.$TGT
FULL_VOCAB=$PROC_PATH/precalc_vocabs/vocab.$SRC-$TGT

echo $TOOLS_PATH
echo $FASTBPE
echo $BPE_CODES
echo $SRC_VOCAB
echo $TGT_VOCAB
echo $FULL_VOCAB

tokenize() {
  lg=$1
  CORPORA_LG_STAGE_RAW=$2
  CORPORA_LG_STAGE_TOK=$3
  if [[ $lg = en ]]; then
    echo "Using EN tokenizer..."
    eval "cat $CORPORA_LG_STAGE_RAW | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lg | $REM_NON_PRINT_CHAR | \
         $TOKENIZER -no-escape -threads $N_THREADS -l $lg > $CORPORA_LG_STAGE_TOK"
  else
    echo "Using ZH tokenizer..."
    eval "cat $CORPORA_LG_STAGE_RAW | $TOOLS_PATH/stanford-segmenter-*/segment.sh pku /dev/stdin UTF-8 0 | \
         $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lg | $REM_NON_PRINT_CHAR > $CORPORA_LG_STAGE_TOK"
  fi
}

for corpora in raw_clean mono para; do
  for stage in train valid test; do
    if [[ $corpora = raw_clean ]]; then
      CORP_PREFIX=para_
      PROC_PREFIX=$stage.raw
    elif [[ $corpora = mono ]]; then
      CORP_PREFIX=''
      PROC_PREFIX=$stage
    elif [[ $corpora = para ]]; then
      CORP_PREFIX=''
      PROC_PREFIX=$stage.$SRC-$TGT
    else
      echo "Corpora $corpora NOT FOUND"
    fi
    for lg in $SRC $TGT; do
      echo "[===============  $corpora - $stage - $lg  ===============]"

      CORPORA_LG_STAGE_RAW=$DATA_PATH/$corpora/$CORP_PREFIX$stage.$lg
      CORPORA_LG_STAGE_BPE=$PROC_PATH/$PROC_PREFIX.$lg
      CORPORA_LG_STAGE_TOK=$CORPORA_LG_STAGE_BPE.tok

      echo $CORPORA_LG_STAGE_RAW
      echo $CORPORA_LG_STAGE_BPE
      echo $CORPORA_LG_STAGE_TOK

      echo "[===================== TOKENIZE =====================]"
      echo "SOURCE FILE: $CORPORA_LG_STAGE_RAW"
      echo "NUMBER OF LINES IN SOURCE FILE: $(wc -l < $CORPORA_LG_STAGE_RAW)"
      if ! [[ -f "$CORPORA_LG_STAGE_TOK" ]]; then
        echo "Tokenizing $corpora - $stage - $lg ..."
        tokenize $lg $CORPORA_LG_STAGE_RAW $CORPORA_LG_STAGE_TOK
      else
        echo "Already tokenized $corpora - $stage - $lg ..."
      fi
      echo "DESTINATION FILE: $CORPORA_LG_STAGE_TOK"
      echo "NUMBER OF LINES IN DESTINATION FILE: $(wc -l < $CORPORA_LG_STAGE_TOK)"
      echo "[===================== TOKENIZE =====================]"

      echo "[================== APPLY BPE CODES =================]"
      echo "SOURCE FILE: $CORPORA_LG_STAGE_TOK"
      echo "NUMBER OF LINES IN SOURCE FILE: $(wc -l < $CORPORA_LG_STAGE_RAW)"
      if ! [[ -f "$CORPORA_LG_STAGE_BPE" ]]; then
        echo "Applying BPE codes $corpora - $stage - $lg ..."
        $FASTBPE applybpe $CORPORA_LG_STAGE_BPE $CORPORA_LG_STAGE_TOK $BPE_CODES
      else
        echo "Already applied BPE codes $corpora - $stage - $lg"
      fi
      echo "DESTINATION FILE: $CORPORA_LG_STAGE_BPE"
      echo "NUMBER OF LINES IN DESTINATION FILE: $(wc -l < $CORPORA_LG_STAGE_BPE)"
      echo "[================== APPLY BPE CODES =================]"

      echo "[===================== BINARIZE =====================]"
      # binarize data
      if ! [[ -f "$CORPORA_LG_STAGE_BPE.pth" ]]; then
        echo "Binarizing $corpora - $stage - $lg ..."
        python3 $MAIN_PATH/preprocess.py $FULL_VOCAB $CORPORA_LG_STAGE_BPE
      else
        echo "Already binarized $corpora - $stage - $lg"
      fi
      echo "[===================== BINARIZE =====================]"
      echo "[====================================================]"
    done
  done
done

for lg in $SRC $TGT; do
  fairseq-preprocess \
  --task cross_lingual_lm \
  --srcdict $PROC_PATH/precalc_vocabs/vocab.$lg \
  --only-source \
  --trainpref $PROC_PATH/train \
  --validpref $PROC_PATH/valid \
  --testpref $PROC_PATH/test \
  --destdir $PROC_PATH \
  --workers 20 \
  --source-lang $lg

  for stage in train valid test
  do
    mv $PROC_PATH/$stage.$lg-None.$lg.bin $PROC_PATH/$stage.$lg.bin
    mv $PROC_PATH/$stage.$lg-None.$lg.idx $PROC_PATH/$stage.$lg.idx
  done
done

fairseq-preprocess \
  --user-dir $MAIN_PATH/mass \
  --task xmasked_seq2seq \
  --source-lang $SRC --target-lang $TGT \
  --trainpref $PROC_PATH/train.$SRC-$TGT \
  --validpref $PROC_PATH/valid.$SRC-$TGT \
  --testpref $PROC_PATH/test.$SRC-$TGT \
  --destdir $PROC_PATH \
  --srcdict $SRC_VOCAB \
  --tgtdict $TGT_VOCAB

echo "DONE!!"
