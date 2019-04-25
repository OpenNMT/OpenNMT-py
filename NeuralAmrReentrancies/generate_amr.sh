#!/bin/bash

GPU_ID=1

VOCAB_DIR=vocab/20M-vocab-gen
MODEL_FILE=models/20M_gen/amr-model_pretrain_epoch19_fine_tune_epoch21.00_4.69.t7
AMR_NL_ALIGN=resources/training-amr-nl-alignments.txt
# Options are: full (normal AMR), stripped (no brackets around leaf nodes, simpler NE and date format), anonymized (like stripped but with NEs, and dates anonymized) 
# Examples:
# full : (w / write :arg1 (s / something))
# stripped :  write :arg1 something
# anonymized : write :arg0 person_name_0 :arg1 something

INPUT_TYPE="${1:-full}"
INPUT_FILE="${2:-resources/sample-data/sample-amr.txt}"
# NOTE: leave output_file same as src_file in order for anonymization/deAnonymization to work properly.
# You may change it safely, if you are inputting already anonymized AMR graphs.

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
