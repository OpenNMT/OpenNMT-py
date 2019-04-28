#!/bin/bash

GPU_ID="${2:-0}"
DATASET=test

MODELS_DIR="${1:-models/graph_gcn_seq}"
INPUT_DATA=ldc2015e86
OUT_DIR=${MODELS_DIR}/preds/
MODEL=${MODELS_DIR}/best.pt

REF_FILE=${INPUT_DATA}/${DATASET}-dfs-linear_src.txt
TARG_FILE=${INPUT_DATA}/${DATASET}-dfs-linear_targ.txt

mkdir -p ${OUT_DIR}
for fname in ${MODELS_DIR}/*model*; do
    f=${fname##*/}
    echo $f
    python translate.py \
	-data_type amr \
        -reentrancies \
        -model ${fname} \
        -src ${REF_FILE} \
        -tgt ${TARG_FILE} \
        -output ${OUT_DIR}/{$f}.pred.txt \
        -beam_size 5 \
        -batch_size 1 \
        -replace_unk \
        -gpu ${GPU_ID} \
        -max_length 125 > log.txt
done
