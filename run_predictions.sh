INCLUDE_TRAIN=${1:-false}
SAVE_PATH=${2:-bt/model}

MAIN_PATH=$PWD
MODEL=$MAIN_PATH/models/checkpoint_best.pt
DATA_DIR=$MAIN_PATH/data/processed
DEST_DIR=$MAIN_PATH/data/$SAVE_PATH
USER_DIR=$MAIN_PATH/mass
TEMP_DIR=$MAIN_PATH/data/tmp

mkdir $DEST_DIR

predict() {
  STAGE=$1
  SRC=$2
  TGT=$3
  SRC_PREFIX=$DATA_DIR/$STAGE.raw
  DEST_PREFIX=$DEST_DIR/$STAGE.$SRC-$TGT
  if ! [[ -f "$TEMP_DIR/$STAGE.$SRC-$TGT.preds" ]]; then
    echo "PREDICTING: $SRC $TGT $STAGE"
    echo "From $SRC_PREFIX to $DEST_PREFIX..."
    start=`date +%s`
    fairseq-interactive $DATA_DIR --user-dir $USER_DIR \
        --fp16 \
        -s $SRC -t $TGT \
        --langs $SRC,$TGT \
        --source-langs $SRC --target-langs $TGT \
        --mt_steps $SRC-$TGT \
        --task xmasked_seq2seq \
        --path $MODEL \
        --beam 8 --remove-bpe  \
        --max-tokens 20000 --buffer-size 600 \
        --input $SRC_PREFIX.$SRC | tee $TEMP_DIR/$STAGE.$SRC-$TGT.preds
    end=`date +%s`
    echo "Runtime: $((end-start))"
  fi
  grep ^H $TEMP_DIR/$STAGE.$SRC-$TGT.preds | cut -f3- | \
    python3 $MAIN_PATH/tools/postprocess_cleanup.py > $DEST_PREFIX.$TGT
  grep ^S $TEMP_DIR/$STAGE.$SRC-$TGT.preds | cut -f2- | \
    python3 $MAIN_PATH/tools/postprocess_cleanup.py > $DEST_PREFIX.$SRC
  echo "[DONE] PREDICTING: $SRC $TGT $STAGE"
  echo "Source sentences saved in $DEST_PREFIX.$SRC"
  echo "Translated sentences saved in $DEST_PREFIX.$TGT"
  echo "[==============================]"
  sleep 5
}

stages=(test)
if [[ $INCLUDE_TRAIN = true ]]; then
  stages+=" train"
fi

echo $stages

for stage in $stages; do
  for lg_src in en zh; do
    if [[ $stage = test && $lg_src = en ]]; then
      continue
    fi
    [[ $lg_src = en ]] && lg_tgt=zh || lg_tgt=en
    predict $stage $lg_src $lg_tgt
  done
done

#grep ^H $RESULTS_DIR/preds.log | cut -f3- > $RESULTS_DIR/preds_.txt
#grep ^H $RESULTS_DIR/preds.log | cut -f1 | cut -c3- | xargs printf "%06d\n" > $RESULTS_DIR/idx.txt
#paste $RESULTS_DIR/idx.txt $RESULTS_DIR/preds_.txt | sort -k1 -n | cut -f2- | \
#  python3 tools/postprocess_cleanup.py > $RESULTS_DIR/preds.txt
#rm $RESULTS_DIR/idx.txt
#rm $RESULTS_DIR/preds_.txt

echo "Done!"
