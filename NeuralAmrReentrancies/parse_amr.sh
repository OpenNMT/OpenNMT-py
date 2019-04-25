#!/bin/bash

GPU_ID=3

VOCAB_DIR=vocab/20M-vocab-parse
MODEL_FILE=models/20M_parse/amr-model_pretrain_epoch4_fine_tune_epoch49.00_2.76.t7
AMR_NL_ALIGN=resources/training-amr-nl-alignments.txt

# Options are: text (that will get anonymized automatically) and textAnonymized (that will bypass any anonymization process)
INPUT_TYPE="${1:-text}"
INPUT_FILE="${2:-resources/sample-data/sample-nl.txt}"
# NOTE: leave output_file same as src_file in order for anonymization/deAnonymization to work properly.
# You may change it safely, if you are inputting already anonymized AMR graphs.


./nerServer.sh 4444&

th evaluate.lua \
	-interactive_mode 0 \
	-model ${MODEL_FILE} \
	-input_type ${INPUT_TYPE} \
	-src_file ${INPUT_FILE} \
	-output_file ${INPUT_FILE} \
	-gpuid ${GPU_ID} \
	-src_dict ${VOCAB_DIR}/amr.src.dict \
	-targ_dict ${VOCAB_DIR}/amr.targ.dict \
	-beam 5 \
	-replace_unk 1 \
	-srctarg_dict ${AMR_NL_ALIGN} \
	-max_sent_l 507 \
	-verbose 0
