#!/bin/bash

GPUIDS="${1:-1}"
DATA_DIR="${2:-amr_reent}"
OUT_DIR="${3:-models/graph_gcn}"
BATCH_SIZE="${4:-100}"
NUM_LAYERS="${5:-1}"

mkdir -p ${OUT_DIR}

python train.py \
    -activation 'relu' \
    -highway 'tanh' \
    -n_gcn_layer 2 \
    -gcn_edge_dropout 0.1 \
    -gcn_dropout 0.1 \
    -data_type amr \
    -data data/${DATA_DIR}/amr \
    -save_model ${OUT_DIR}/amr-model \
    -layers ${NUM_LAYERS} \
    -report_every 50 \
    -train_steps 20001 \
    -valid_steps 150 \
    -rnn_size 750 \
    -word_vec_size 750 \
    -gcn_vec_size 750 \
    -encoder_type gcn \
    -decoder_type rnn \
    -batch_size ${BATCH_SIZE} \
    -max_generator_batches 50 \
    -save_checkpoint_steps 2500 \
    -decay_steps 150 \
    -optim sgd \
    -max_grad_norm 3 \
    -learning_rate_decay 0.8 \
    -start_decay_steps 6000 \
    -learning_rate 1 \
    -dropout 0.5 \
    -gpu_ranks ${GPUIDS} \
    -seed 123 
