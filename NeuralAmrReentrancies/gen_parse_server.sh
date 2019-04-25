#!/bin/bash

GPU_ID=3

VOCAB_PARSE_DIR=vocab/20M-vocab-parse
VOCAB_GEN_DIR=vocab/20M-vocab-gen
MODEL_PARSE_FILE=models/20M_parse/amr-model_pretrain_epoch4_fine_tune_epoch49.00_2.76.t7
MODEL_GEN_FILE=models/20M_gen/amr-model_pretrain_epoch19_fine_tune_epoch21.00_4.69.t7
AMR_NL_ALIGN=resources/training-amr-nl-alignments.txt

./nerServer.sh 4444&

# GENERATOR
th evaluate_server.lua \
	-model ${MODEL_GEN_FILE} \
	-gpuid ${GPU_ID} \
	-src_dict ${VOCAB_GEN_DIR}/amr.src.dict \
	-targ_dict ${VOCAB_GEN_DIR}/amr.targ.dict \
	-beam 5 \
	-replace_unk 1 \
	-srctarg_dict ${AMR_NL_ALIGN} \
	-port 4447 \
	-verbose 1&

# PARSER
th evaluate_server.lua \
	-model ${MODEL_PARSE_FILE} \
	-gpuid ${GPU_ID} \
	-src_dict ${VOCAB_PARSE_DIR}/amr.src.dict \
	-targ_dict ${VOCAB_PARSE_DIR}/amr.targ.dict \
	-beam 5 \
	-replace_unk 1 \
	-srctarg_dict ${AMR_NL_ALIGN} \
	-port 4448\
	-verbose 1

