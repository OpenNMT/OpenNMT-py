#!/bin/bash

GPUIDS="${1:-0}"
DATA_DIR="${2:-amr}"
OUT_DIR="${3:-models/seq}"
BATCH_SIZE="${4:-100}"
NUM_LAYERS="${5:-1}"

mkdir -p ${OUT_DIR}

python train.py \
    -data_type 'amr' \
    -data data/${DATA_DIR}/amr \
    -save_model ${OUT_DIR}/amr-model \
    -layers ${NUM_LAYERS} \
    -report_every 50 \
    -train_steps 20001 \
    -valid_steps 150 \
    -rnn_size 900 \
    -word_vec_size 450 \
    -encoder_type brnn \
    -decoder_type rnn \
    -batch_size ${BATCH_SIZE} \
    -max_generator_batches 50 \
    -learning_rate_decay 0.8 \
    -start_decay_steps 6000 \
    -save_checkpoint_steps 2500 \
    -decay_steps 150 \
    -optim sgd \
    -max_grad_norm 3 \
    -learning_rate 1 \
    -seed 123 \
    -dropout 0.5 \
    -gpu_ranks ${GPUIDS}
