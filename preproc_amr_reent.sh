#!/bin/bash

OUT_DIR=data/"${1:-amr_reent}"
DATA_DIR="ldc2017t10"

mkdir -p ${OUT_DIR}

python preprocess.py \
    -seed 123 \
    -reentrancies \
    -data_type amr \
    -train_src ${DATA_DIR}/training-dfs-linear_src.txt \
    -train_tgt ${DATA_DIR}/training-dfs-linear_targ.txt \
    -valid_src ${DATA_DIR}/dev-dfs-linear_src.txt \
    -valid_tgt ${DATA_DIR}/dev-dfs-linear_targ.txt \
    -save_data ${OUT_DIR}/amr \
    -src_words_min_frequency 1 \
    -tgt_words_min_frequency 1 \
    -src_seq_length 125 \
    -tgt_seq_length 125 \
    -src_vocab dicts/amr.src.dict \
    -tgt_vocab dicts/amr.targ.dict
#dicts taken from https://github.com/sinantie/NeuralAmr
