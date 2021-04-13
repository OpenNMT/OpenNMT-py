#!/bin/bash

##################################################################################
# The default script downloads the commoncrawl, europarl and newstest2014 and
# newstest2017 datasets. Files that are not English or German are removed in
# this script for tidyness.You may switch datasets out depending on task.
# (Note that commoncrawl europarl-v7 are the same for all tasks).
# http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
# http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
#
# WMT14 http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz
# WMT15 http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz
# WMT16 http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz
# WMT17 http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz
# Note : there are very little difference, but each year added a few sentences
# new WMT17 http://data.statmt.org/wmt17/translation-task/rapid2016.tgz
#
# For WMT16 Rico Sennrich released some News back translation
# http://data.statmt.org/rsennrich/wmt16_backtranslations/en-de/
#
# Tests sets: http://data.statmt.org/wmt17/translation-task/test.tgz
##################################################################################

# provide script usage instructions
if [ $# -eq 0 ]
then
    echo "usage: $0 <data_dir>"
    exit 1
fi

# set relevant paths
SP_PATH=/usr/local/bin
DATA_PATH=$1
TEST_PATH=$DATA_PATH/test

CUR_DIR=$(pwd)

# set vocabulary size and source and target languages
vocab_size=32000
sl=en
tl=de

# Download the default datasets into the $DATA_PATH; mkdir if it doesn't exist
mkdir -p $DATA_PATH
cd $DATA_PATH

echo "Downloading and extracting Commoncrawl data (919 MB) for training..."
wget --trust-server-names http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
tar zxvf training-parallel-commoncrawl.tgz
ls | grep -v 'commoncrawl.de-en.[de,en]' | xargs rm

echo "Downloading and extracting Europarl data (658 MB) for training..."
wget --trust-server-names http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
tar zxvf training-parallel-europarl-v7.tgz
cd training && ls | grep -v 'europarl-v7.de-en.[de,en]' | xargs rm
cd .. && mv training/europarl* . && rm -r training training-parallel-europarl-v7.tgz

echo "Downloading and extracting News Commentary data (76 MB) for training..."
wget --trust-server-names http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz
tar zxvf training-parallel-nc-v11.tgz
cd training-parallel-nc-v11 && ls | grep -v news-commentary-v11.de-en.[de,en] | xargs rm
cd .. && mv training-parallel-nc-v11/* . && rm -r training-parallel-nc-v11 training-parallel-nc-v11.tgz

# Validation and test data are put into the $DATA_PATH/test folder
echo "Downloading and extracting newstest2014 data (4 MB) for validation..."
wget --trust-server-names http://www.statmt.org/wmt14/test-filtered.tgz
echo "Downloading and extracting newstest2017 data (5 MB) for testing..."
wget --trust-server-names http://data.statmt.org/wmt17/translation-task/test.tgz
tar zxvf test-filtered.tgz && tar zxvf test.tgz
cd test && ls | grep -v '.*deen\|.*ende' | xargs rm
cd .. && rm test-filtered.tgz test.tgz && cd ..

# set training, validation, and test corpuses
corpus[1]=commoncrawl.de-en
corpus[2]=europarl-v7.de-en
corpus[3]=news-commentary-v11.de-en
#corpus[3]=news-commentary-v12.de-en
#corpus[4]=news.bt.en-de
#corpus[5]=rapid2016.de-en

validset=newstest2014-deen
testset=newstest2017-ende

cd $CUR_DIR

# retrieve file preparation from Moses repository
wget -nc \
    https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/ems/support/input-from-sgm.perl \
    -O $TEST_PATH/input-from-sgm.perl

##################################################################################
# Starting from here, original files are supposed to be in $DATA_PATH
# a data folder will be created in scripts/wmt
##################################################################################

export PATH=$SP_PATH:$PATH

# Data preparation using SentencePiece
# First we concat all the datasets to train the SP model
if true; then
 echo "$0: Training sentencepiece model"
 rm -f $DATA_PATH/train.txt
 for ((i=1; i<= ${#corpus[@]}; i++))
 do
  for f in $DATA_PATH/${corpus[$i]}.$sl $DATA_PATH/${corpus[$i]}.$tl
   do
    cat $f >> $DATA_PATH/train.txt
   done
 done
 spm_train --input=$DATA_PATH/train.txt --model_prefix=$DATA_PATH/wmt$sl$tl \
           --vocab_size=$vocab_size --character_coverage=1
 rm $DATA_PATH/train.txt
fi

# Second we use the trained model to tokenize all the files
# This is not necessary, as it can be done on the fly in OpenNMT-py 2.0
# if false; then
#  echo "$0: Tokenizing with sentencepiece model"
#  rm -f $DATA_PATH/train.txt
#  for ((i=1; i<= ${#corpus[@]}; i++))
#  do
#   for f in $DATA_PATH/${corpus[$i]}.$sl $DATA_PATH/${corpus[$i]}.$tl
#    do
#     file=$(basename $f)
#     spm_encode --model=$DATA_PATH/wmt$sl$tl.model < $f > $DATA_PATH/$file.sp
#    done
#  done
# fi

# We concat the training sets into two (src/tgt) tokenized files
# if false; then
#  cat $DATA_PATH/*.$sl.sp > $DATA_PATH/train.$sl
#  cat $DATA_PATH/*.$tl.sp > $DATA_PATH/train.$tl
# fi

#  We use the same tokenization method for a valid set (and test set)
# if true; then
#  perl $TEST_PATH/input-from-sgm.perl < $TEST_PATH/$validset-src.$sl.sgm \
#     | spm_encode --model=$DATA_PATH/wmt$sl$tl.model > $DATA_PATH/valid.$sl.sp
#  perl $TEST_PATH/input-from-sgm.perl < $TEST_PATH/$validset-ref.$tl.sgm \
#     | spm_encode --model=$DATA_PATH/wmt$sl$tl.model > $DATA_PATH/valid.$tl.sp
#  perl $TEST_PATH/input-from-sgm.perl < $TEST_PATH/$testset-src.$sl.sgm \
#     | spm_encode --model=$DATA_PATH/wmt$sl$tl.model > $DATA_PATH/test.$sl.sp
#  perl $TEST_PATH/input-from-sgm.perl < $TEST_PATH/$testset-ref.$tl.sgm \
#     | spm_encode --model=$DATA_PATH/wmt$sl$tl.model > $DATA_PATH/test.$tl.sp
# fi

# Parse the valid and test sets
if true; then
 perl $TEST_PATH/input-from-sgm.perl < $TEST_PATH/$validset-src.$sl.sgm \
    > $DATA_PATH/valid.$sl
 perl $TEST_PATH/input-from-sgm.perl < $TEST_PATH/$validset-ref.$tl.sgm \
    > $DATA_PATH/valid.$tl
 perl $TEST_PATH/input-from-sgm.perl < $TEST_PATH/$testset-src.$sl.sgm \
    > $DATA_PATH/test.$sl
 perl $TEST_PATH/input-from-sgm.perl < $TEST_PATH/$testset-ref.$tl.sgm \
    > $DATA_PATH/test.$tl
fi
