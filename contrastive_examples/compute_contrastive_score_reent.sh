#!/bin/bash

GPU_ID="${3:-0}"
FNAME="${1:-models/graph_gcn_seq/amr-model_step_12500.pt}"
MODELS_DIR=$(dirname "${FNAME}")
INPUT_DATA=ldc2015e86
OUT_DIR=${MODELS_DIR}/contrastive/
MODEL=${MODELS_DIR}/best.pt

REF_FILE="reen_contr.src.txt"
TARG_FILE="reen_contr.ref.txt"

mkdir -p ${OUT_DIR}
f=${fname##*/}

python translate.py \
        -reentrancies \
	-data_type amr \
        -model ${FNAME} \
        -src ${REF_FILE} \
        -tgt ${TARG_FILE} \
        -output ${OUT_DIR}/{$f}.pred_norm.txt \
        -gpu ${GPU_ID} \
        -beam_size 5 \
        -batch_size 1 \
        -replace_unk \
        -max_length 125 > normal_scores.txt

TARG_FILE="reen_contr.contr.txt"
python translate.py \
        -reentrancies \
        -data_type amr \
        -model ${FNAME} \
        -src ${REF_FILE} \
        -tgt ${TARG_FILE} \
        -output ${OUT_DIR}/{$f}.pred_contr.txt \
        -gpu ${GPU_ID} \
        -beam_size 5 \
        -batch_size 1 \
        -replace_unk \
        -max_length 125 > contr_scores.txt

python acc_contrastive.py normal_scores.txt contr_scores.txt
