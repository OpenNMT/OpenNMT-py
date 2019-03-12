#!/bin/bash

TYPE="GEND"
GPU_ID="${3:-0}"

echo "Seq"
FNAME="${1:-models/seq/amr-model_step_12500.pt}"
MODELS_DIR=$(dirname "${FNAME}")
INPUT_DATA=ldc2015e86
OUT_DIR=${MODELS_DIR}/contrastive/
MODEL=${MODELS_DIR}/best.pt

REF_FILE=$TYPE"_contr.src.txt"
TARG_FILE=$TYPE"_contr.ref.txt"

mkdir -p ${OUT_DIR}
f=${FNAME##*/}

python translate.py \
	-data_type amr \
        -model ${FNAME} \
        -src ${REF_FILE} \
        -tgt ${TARG_FILE} \
        -output ${OUT_DIR}/{$f}.pred_norm.txt \
        -gpu ${GPU_ID} \
        -beam_size 5 \
        -batch_size 1 \
        -replace_unk \
        -max_length 125 > normal_scores1.txt

TARG_FILE=$TYPE"_contr.contr.txt"
python translate.py \
        -data_type amr \
        -model ${FNAME} \
        -src ${REF_FILE} \
        -tgt ${TARG_FILE} \
        -output ${OUT_DIR}/{$f}.pred_contr.txt \
        -gpu ${GPU_ID} \
        -beam_size 5 \
        -batch_size 1 \
        -replace_unk \
        -max_length 125 > contr_scores1.txt

python acc_contrastive.py normal_scores1.txt contr_scores1.txt

echo "GCN_EMB (tree)"
FNAME="${1:-models/gcn_emb_ldc2017t10_seed4/amr-model_step_12500.pt}"
MODELS_DIR=$(dirname "${FNAME}")

INPUT_DATA=ldc2017t10
OUT_DIR=${MODELS_DIR}/contrastive/
MODEL=${MODELS_DIR}/best.pt

REF_FILE=$TYPE"_contr.src.txt"
TARG_FILE=$TYPE"_contr.ref.txt"

mkdir -p ${OUT_DIR}
f=${FNAME##*/}

python translate.py \
        -data_type amr \
        -model ${FNAME} \
        -src ${REF_FILE} \
        -tgt ${TARG_FILE} \
        -output ${OUT_DIR}/{$f}.pred_norm.txt \
        -gpu ${GPU_ID} \
        -beam_size 5 \
        -batch_size 1 \
        -replace_unk \
        -max_length 125 > normal_scores2.txt

TARG_FILE=$TYPE"_contr.contr.txt"
python translate.py \
        -data_type amr \
        -model ${FNAME} \
        -src ${REF_FILE} \
        -tgt ${TARG_FILE} \
        -output ${OUT_DIR}/{$f}.pred_contr.txt \
        -gpu ${GPU_ID} \
        -beam_size 5 \
        -batch_size 1 \
        -replace_unk \
        -max_length 125 > contr_scores2.txt

python acc_contrastive.py normal_scores2.txt contr_scores2.txt

echo "GCN_EMB (graph)"
FNAME="${1:-models/gcn_emb_reent_ldc2017t10_seed4/amr-model_step_10000.pt}"
MODELS_DIR=$(dirname "${FNAME}")

INPUT_DATA=ldc2017t10
OUT_DIR=${MODELS_DIR}/contrastive/
MODEL=${MODELS_DIR}/best.pt

REF_FILE=$TYPE"_contr.src.txt"
TARG_FILE=$TYPE"_contr.ref.txt"

mkdir -p ${OUT_DIR}
f=${FNAME##*/}

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
        -max_length 125 > normal_scores3.txt

TARG_FILE=$TYPE"_contr.contr.txt"
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
        -max_length 125 > contr_scores3.txt

python acc_contrastive.py normal_scores3.txt contr_scores3.txt
