#!/bin/bash

##################################################################################
# This script will download wikitext-103-raw and will do basic data preparation
# for BPE and training
##################################################################################

# provide script usage instructions
if [ $# -eq 0 ]
then
    echo "usage: $0 <data_dir>"
    exit 1
fi

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

# set relevant paths
SP_PATH=/usr/local/bin
DATA_PATH=$1
TEST_PATH=$DATA_PATH/test

CUR_DIR=$(pwd)

# Download the default datasets into the $DATA_PATH; mkdir if it doesn't exist
mkdir -p $DATA_PATH
cd $DATA_PATH

echo "Downloading and extracting WikiText-103 (183 MB) for training and inference..."
wget --trust-server-names https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
rm wikitext-103-raw-v1.zip
cd wikitext-103-raw

echo "Removing empty lines and shuffling training data"
sed -r '/^\s*$/d' -i wiki.train.raw
sed -r '/^\s*$/d' -i wiki.valid.raw
sed -r '/^\s*$/d' -i wiki.test.raw
sort --random-source=<(get_seeded_random 42) -R -o wiki.train.raw wiki.train.raw
