SRC=en
TGT=zh

MAIN_PATH=$PWD
RAW_PATH=$MAIN_PATH/data/raw
BT_PATH=$MAIN_PATH/data/bt/${1:-model}
TMP_PATH=$MAIN_PATH/data/tmp
MONO_PATH=$MAIN_PATH/data/mono
PARA_PATH=$MAIN_PATH/data/para
CLEAN_PATH=$MAIN_PATH/data/raw_clean
PROC_PATH=$MAIN_PATH/data/processed

TOOLS_PATH=$MAIN_PATH/tools
CLEANUP="python3 $TOOLS_PATH/preprocess_cleanup.py"

echo $MAIN_PATH
echo $RAW_PATH
echo $BT_PATH
echo $TOOLS_PATH

MONO_SRC_ALL=$RAW_PATH/train_$SRC.csv
MONO_TGT_ALL=$RAW_PATH/train_$TGT.csv

echo $MONO_SRC_ALL
echo $MONO_TGT_ALL

get_seeded_random() {
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

echo "[===============  CLEAN  ===============]"
echo "Cleaning up raw files..."

if ! [[ -f "$CLEAN_PATH/mono_train.$SRC" && -f "$CLEAN_PATH/mono_train.$TGT" ]]; then
  echo "Cleaning up mono_train..."
  for lg in $SRC $TGT; do
    cat $RAW_PATH/train_$lg.csv | $CLEANUP > $CLEAN_PATH/mono_train.$lg
  done
fi
echo "Cleaned up mono_train in: $CLEAN_PATH/mono_train.$lg"

if ! [[ -f "$CLEAN_PATH/para_train.$SRC" && -f "$CLEAN_PATH/para_train.$TGT" ]]; then
  echo "Cleaning up para_train..."
  for lg in $SRC $TGT; do
    cat $BT_PATH/train.$SRC-$TGT.$lg $BT_PATH/train.$TGT-$SRC.$lg | \
      $CLEANUP > $CLEAN_PATH/para_train.$lg
  done
fi
echo "Cleaned up para_train in: $CLEAN_PATH/para_train.$lg"

if ! [[ -f "$CLEAN_PATH/para_valid.$SRC" && -f "$CLEAN_PATH/para_valid.$TGT" ]]; then
  echo "Cleaning up para_valid..."
  for lg in $SRC $TGT; do
    cat $RAW_PATH/dev_$lg.csv | $CLEANUP > $CLEAN_PATH/para_valid.$lg
  done
fi
echo "Cleaned up para_valid in: $CLEAN_PATH/para_valid.$lg"

if ! [[ -f "$CLEAN_PATH/para_test.$SRC" && -f "$CLEAN_PATH/para_test.$TGT" ]]; then
  echo "Cleaning up para_test..."
  cat $BT_PATH/test.$TGT-$SRC.$SRC | $CLEANUP > $CLEAN_PATH/para_test.$SRC
  cat $RAW_PATH/test_$TGT.csv      | $CLEANUP > $CLEAN_PATH/para_test.$TGT
fi
echo "Cleaned up para_test in: $CLEAN_PATH/para_test.$lg"

echo "Clean mono_train sizes: $(wc -l < $CLEAN_PATH/mono_train.$SRC) ; $(wc -l < $CLEAN_PATH/mono_train.$TGT)"
echo "Clean para_train sizes:   $(wc -l < $CLEAN_PATH/para_train.$SRC) = $(wc -l < $CLEAN_PATH/para_train.$TGT)"
echo "Clean para_valid sizes:   $(wc -l < $CLEAN_PATH/para_valid.$SRC) = $(wc -l < $CLEAN_PATH/para_valid.$TGT)"
echo "Clean para_test sizes:   $(wc -l < $CLEAN_PATH/para_test.$SRC) = $(wc -l < $CLEAN_PATH/para_test.$TGT)"
echo "[===============  CLEAN  ===============]"

echo "[===============  MONO  ===============]"
echo "Train-valid-test split monolingual files..."
for lg in $SRC $TGT; do
  if ! [[ -f "$MONO_PATH/train.$lg" && -f "$MONO_PATH/valid.$lg" && -f "$MONO_PATH/test.$lg" ]]; then
    MONO_FILE=$CLEAN_PATH/mono_train.$lg
    shuf $MONO_FILE --random-source=<(get_seeded_random 42) | \
      split -a1 -d -l $(( $(wc -l < $MONO_FILE) - 20000 )) - $TMP_PATH/train_test_split
    shuf $TMP_PATH/train_test_split1 --random-source=<(get_seeded_random 42) | \
      split -a1 -d -l 10000 - $TMP_PATH/valid_test_split
    mv $TMP_PATH/train_test_split0 $MONO_PATH/train.$lg
    mv $TMP_PATH/valid_test_split0 $MONO_PATH/valid.$lg
    mv $TMP_PATH/valid_test_split1 $MONO_PATH/test.$lg
    rm $TMP_PATH/train_test_split1
  fi
done

cp $PROC_PATH/precalc_vocabs/vocab.$SRC $MONO_PATH/vocab.$SRC
cp $PROC_PATH/precalc_vocabs/vocab.$TGT $MONO_PATH/vocab.$TGT

echo "Mono train sizes:  $(wc -l < $MONO_PATH/train.$SRC) ; $(wc -l < $MONO_PATH/train.$TGT)"
echo "Mono valid sizes:   $(wc -l < $MONO_PATH/valid.$SRC) ; $(wc -l < $MONO_PATH/valid.$TGT)"
echo "Mono test sizes:    $(wc -l < $MONO_PATH/test.$SRC) ; $(wc -l < $MONO_PATH/test.$TGT)"
echo "[===============  MONO  ===============]"

echo "[===============  PARA  ===============]"
mkdir $PARA_PATH/raw/
echo "Train-valid-test split parallel files..."
for lg in $SRC $TGT; do
  echo "Splitting $lg files..."
  if ! [[ -f "$PARA_PATH/raw/train.$lg" && \
          -f "$PARA_PATH/raw/valid.$lg" && \
          -f "$PARA_PATH/raw/test.$lg" ]]; then
    PARA_FILE=$TMP_PATH/all.$lg
    echo "Parallel test file: $TEST_FILE"
    cat $CLEAN_PATH/para_train.$lg $CLEAN_PATH/para_valid.$lg $CLEAN_PATH/para_test.$lg > $PARA_FILE
    shuf $PARA_FILE --random-source=<(get_seeded_random 42) | \
      split -a1 -d -l $(( $(wc -l < $PARA_FILE) - 10000 )) - $TMP_PATH/train_test_split
    shuf $TMP_PATH/train_test_split1 --random-source=<(get_seeded_random 42) | \
      split -a1 -d -l 5000 - $TMP_PATH/valid_test_split
    mv $TMP_PATH/train_test_split0 $PARA_PATH/raw/train.$lg
    mv $TMP_PATH/valid_test_split0 $PARA_PATH/raw/valid.$lg
    mv $TMP_PATH/valid_test_split1 $PARA_PATH/raw/test.$lg
    rm $TMP_PATH/train_test_split1
    rm $PARA_FILE
  fi
  echo "Done splitting $lg files"
done

if ! [[ -f "$PARA_PATH/train.$lg" && \
        -f "$PARA_PATH/valid.$lg" && \
        -f "$PARA_PATH/test.$lg" ]]; then
  for stage in test valid train; do
    echo "stage: $stage"
    res=($(grep -n '^$' $PARA_PATH/raw/$stage.$SRC | rev | cut -c2- | rev))
    res+=($(grep -n '^$' $PARA_PATH/raw/$stage.$TGT | rev | cut -c2- | rev))
    echo "empty lines: ${res[*]}"

    for lg in en zh; do
      echo "deleting empty lines in $PARA_PATH/$stage.$lg"
      sed -i ''"${res[*]/%/d;}"'' $PARA_PATH/raw/$stage.$lg
      cp $PARA_PATH/raw/$stage.$lg $PARA_PATH/$stage.$lg
      echo "deleted empty lines in $PARA_PATH/$stage.$lg"
    done
  done
fi

cp $PROC_PATH/precalc_vocabs/vocab.$SRC $PARA_PATH/vocab.$SRC
cp $PROC_PATH/precalc_vocabs/vocab.$TGT $PARA_PATH/vocab.$TGT

echo "Para train sizes:   $(wc -l < $PARA_PATH/train.$SRC) = $(wc -l < $PARA_PATH/train.$TGT)"
echo "Para valid sizes:    $(wc -l < $PARA_PATH/valid.$SRC) = $(wc -l < $PARA_PATH/valid.$TGT)"
echo "Para test sizes:     $(wc -l < $PARA_PATH/test.$SRC) = $(wc -l < $PARA_PATH/test.$TGT)"
echo "[===============  PARA  ===============]"

echo ""
echo "Generated files:"
for corpora in raw_clean mono para; do
  for stage in train valid test; do
    if [[ $corpora = raw_clean ]]; then
      PROC_PREFIX=$stage.raw
    elif [[ $corpora = mono ]]; then
      PROC_PREFIX=$stage
    elif [[ $corpora = para ]]; then
      PROC_PREFIX=$stage.$SRC-$TGT
    else
      echo "Corpora $corpora NOT FOUND"
    fi
    for lg in $SRC $TGT; do
      echo "$corpora $stage $lg"
    done
  done
done
