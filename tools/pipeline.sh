#!/usr/bin/env bash
# Author : Thamme Gowda
# Created : Nov 06, 2017

ONMT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

#======= EXPERIMENT SETUP ======
# Activate python environment if needed
# source ~/.bashrc
# source activate py3

# update these variables
NAME="run1"
OUT="./onmt-runs"

DATA=""
TRAIN_SRC=$DATA/*train.src
TRAIN_TGT=$DATA/*train.tgt
VALID_SRC=$DATA/*dev.src
VALID_TGT=$DATA/*dev.tgt
TEST_SRC=$DATA/*test.src
TEST_TGT=$DATA/*test.tgt

BPE="" # default
BPE="src+tgt" # src, tgt, src+tgt

# applicable only when BPE="src" or "src+tgt"
BPE_SRC_OPS=10000

# applicable only when BPE="tgt" or "src+tgt"
BPE_TGT_OPS=10000

GPUARG="" # default
GPUARG="-gpuid 0"

#====== EXPERIMENT BEGIN ======

echo "Output dir = $OUT"
[ -d $OUT ] || mkdir -p $OUT
[ -d $OUT/data ] || mkdir -p $OUT/data
[ -d $OUT/models ] || mkdir $OUT/models
[ -d $OUT/test ] || mkdir -p  $OUT/test

if [[ "$BPE" == *"src"* ]]; then
    # Here we could use more  monolingual data
    $ONMT/tools/learn_bpe.py -s $BPE_SRC_OPS < $TRAIN_SRC > $OUT/data/bpe-codes.src

    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.src <  $TRAIN_SRC > $OUT/data/train.src
    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.src <  $VALID_SRC > $OUT/data/valid.src
    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.src <  $TEST_SRC > $OUT/data/test.src
else
    ln -s $TRAIN_SRC $OUT/data/train.src
    ln -s $VALID_SRC $OUT/data/valid.src
    ln -s $TEST_SRC $OUT/data/test.src
fi


if [[ "$BPE" == *"tgt"* ]]; then
    # Here we could use more  monolingual data
    $ONMT/tools/learn_bpe.py -s $BPE_SRC_OPS < $TRAIN_TGT > $OUT/data/bpe-codes.tgt

    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.tgt <  $TRAIN_TGT > $OUT/data/train.tgt
    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.tgt <  $VALID_TGT > $OUT/data/valid.tgt
    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.tgt <  $TEST_TGT > $OUT/data/test.tgt
else
    ln -s $TRAIN_TGT $OUT/data/train.tgt
    ln -s $VALID_TGT $OUT/data/valid.tgt
    ln -s $TEST_TGT $OUT/data/test.tgt
fi


# Disable training
#: <<EOF
echo "Step1: preprocess"
python $ONMT/preprocess.py \
    -train_src $OUT/data/train.src \
    -train_tgt $OUT/data/train.tgt \
    -valid_src $OUT/data/valid.src \
    -valid_tgt $OUT/data/valid.tgt \
    -save_data $OUT/data/processed

echo "Step2: Train"
python $ONMT/train.py -data $OUT/data/processed -save_model $OUT/models/$NAME "$GPUARG"

#EOF

# select a model with high accuracy and low perplexity
# TODO: currently using linear scale, maybe not the best
model=`ls $OUT/models/*.pt| awk -F '_' 'BEGIN{maxv=-1000000} {score=$(NF-3)-$(NF-1); if (score > maxv) {maxv=score; max=$0}}  END{ print max}'`
echo "Chosen Model = $model"

echo "Step 3a: Translate Test"
python $ONMT/translate.py -model $model -src $TEST_SRC -output $OUT/test/test.out  -replace_unk  -verbose "$GPUARG" > $OUT/test/test.log

echo "Step 3b: Translate Dev"
python $ONMT/translate.py -model $model -src $DEV_SRC -output $OUT/test/dev.out  -replace_unk -verbose "$GPUARG" > $OUT/test/dev.log

if [[ "$BPE" == *"tgt"* ]]; then
    echo "ERROR:: BPE undo for target is pending. BLEU is incorrect"
fi

echo "Step 4a: Evaluate Test"
$ONMT/tools/multi-bleu-detok.perl $TEST_TGT < $OUT/test/test.out > $OUT/test/test.tc.bleu
$ONMT/tools/multi-bleu-detok.perl -lc $TEST_TGT < $OUT/test/test.out > $OUT/test/test.lc.bleu

echo "Step 4b: Evaluate Dev"
$ONMT/tools/multi-bleu-detok.perl $DEV_TGT < $OUT/test/dev.out > $OUT/test/dev.tc.bleu
$ONMT/tools/multi-bleu-detok.perl -lc $DEV_TGT < $OUT/test/dev.out > $OUT/test/dev.lc.bleu

#===== EXPERIMENT END ======
