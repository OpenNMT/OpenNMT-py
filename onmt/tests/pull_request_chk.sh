#!/bin/bash
# Run this script and fix *any* error before sending PR.
# For repeated runs, set the environment variables
# SKIP_DOWNLOADS=1  If files/uncompressed dirs exist don't download (if compressed files exist, just untar).
# SKIP_FULL_CLEAN=1  Don't remove anything downloaded/uncompressed.

LOG_FILE=/tmp/$$_pull_request_chk.log
echo > ${LOG_FILE} # Empty the log file.

PROJECT_ROOT=`dirname "$0"`"/../../"
DATA_DIR="$PROJECT_ROOT/data"
TEST_DIR="$PROJECT_ROOT/onmt/tests"
PYTHON="python3"
TMP_OUT_DIR="/tmp/onmt_prchk"

clean_up()
{
    if [[ "$1" != "error" ]]; then
        rm ${LOG_FILE}
    fi
    if [[ "${SKIP_FULL_CLEAN}" == "1" ]]; then
        # delete any .pt's that weren't downloaded
        ls $TMP_OUT_DIR/*.pt | xargs -I {} rm -f $TMP_OUT_DIR/{}
    else
        # delete all .pt's
        rm -f $TMP_OUT_DIR/*.pt
        rm -rf $TMP_OUT_DIR/sample
        rm $TMP_OUT_DIR/onmt.vocab*
        rm -d $TMP_OUT_DIR
    fi
}
trap clean_up SIGINT SIGQUIT SIGKILL

error_exit()
{
    echo "Failed !" | tee -a ${LOG_FILE}
    echo "[!] Check ${LOG_FILE} for detail."
    clean_up error
    exit 1
}

# environment_prepare()
# {

# }

# flake8 check
echo -n "[+] Doing flake8 check..."
${PYTHON} -m flake8 >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


# unittest
echo -n "[+] Doing unittest test..."
${PYTHON} -m unittest discover >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


#
# Get Vocabulary test
#
echo -n "[+] Testing vocabulary building..."
PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH} ${PYTHON} onmt/bin/build_vocab.py \
            -config ${DATA_DIR}/data.yaml \
            -save_data $TMP_OUT_DIR/onmt \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.tgt \
            -n_sample 5000 >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -r $TMP_OUT_DIR/sample

#
# Training test
#
echo -n "[+] Testing NMT fields/transforms prepare..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/data.yaml \
            -save_data $TMP_OUT_DIR/onmt.train.check \
            -dump_fields -dump_transforms -n_sample 30 \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
# rm $TMP_OUT_DIR/onmt.train.check*  # used in tool testing

echo "[+] Doing Training test..."

echo -n "  [+] Testing NMT training..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -rnn_size 2 -batch_size 10 \
            -word_vec_size 5 -report_every 5        \
            -rnn_size 10 -train_steps 10 >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing NMT training w/ copy..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -rnn_size 2 -batch_size 10 \
            -word_vec_size 5 -report_every 5        \
            -rnn_size 10 -train_steps 10 \
            -copy_attn >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing NMT training w/ align..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/align_data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -max_generator_batches 0 \
            -encoder_type transformer -decoder_type transformer \
            -layers 4 -word_vec_size 16 -rnn_size 16 -heads 2 -transformer_ff 64 \
            -lambda_align 0.05 -alignment_layer 2 -alignment_heads 0 \
            -report_every 5 -train_steps 10 >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing NMT training w/ coverage..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -rnn_size 2 -batch_size 10 \
            -word_vec_size 5 -report_every 5        \
            -coverage_attn true -lambda_coverage 0.1 \
            -rnn_size 10 -train_steps 10 >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing LM training..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/lm_data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -model_task lm \
            -encoder_type transformer_lm \
            -decoder_type transformer_lm \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -dec_layers 2 -batch_size 10 \
            -heads 4 -transformer_ff 64 \
            -word_vec_size 16 -report_every 5        \
            -rnn_size 16 -train_steps 10 >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing LM training w/ copy..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/lm_data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -model_task lm \
            -encoder_type transformer_lm \
            -decoder_type transformer_lm \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -dec_layers 2 -batch_size 10 \
            -heads 4 -transformer_ff 64 \
            -word_vec_size 16 -report_every 5        \
            -rnn_size 16 -train_steps 10 \
            -copy_attn >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}*
rm $TMP_OUT_DIR/onmt.vocab*

echo -n "  [+] Testing Graph Neural Network training..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/ggnn_data.yaml \
            -src_seq_length 1000 -tgt_seq_length 30 \
            -encoder_type ggnn -layers 2 \
            -decoder_type rnn -rnn_size 256 \
            -learning_rate 0.1 -learning_rate_decay 0.8 \
            -global_attention general -batch_size 32 -word_vec_size 256 \
            -bridge -train_steps 10 -n_edge_types 9 -state_dim 256 \
            -n_steps 10 -n_node 64 >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

#
# Translation test
#
echo "[+] Doing translation test..."

echo -n "  [+] Testing NMT translation..."
head ${DATA_DIR}/src-test.txt > $TMP_OUT_DIR/src-test.txt
${PYTHON} translate.py -model ${TEST_DIR}/test_model.pt -src $TMP_OUT_DIR/src-test.txt -verbose >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/src-test.txt

echo -n "  [+] Testing NMT ensemble translation..."
head ${DATA_DIR}/src-test.txt > $TMP_OUT_DIR/src-test.txt
${PYTHON} translate.py -model ${TEST_DIR}/test_model.pt ${TEST_DIR}/test_model.pt \
            -src $TMP_OUT_DIR/src-test.txt -verbose >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/src-test.txt

echo -n "  [+] Testing NMT translation w/ Beam search..."
${PYTHON} translate.py -model ${TEST_DIR}/test_model2.pt  \
            -src ${DATA_DIR}/morph/src.valid   \
            -verbose -batch_size 10     \
            -beam_size 10 \
            -tgt ${DATA_DIR}/morph/tgt.valid   \
            -out $TMP_OUT_DIR/trans_beam  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/morph/tgt.valid $TMP_OUT_DIR/trans_beam
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/trans_beam

echo -n "  [+] Testing NMT translation w/ Random Sampling..."
${PYTHON} translate.py -model ${TEST_DIR}/test_model2.pt  \
            -src ${DATA_DIR}/morph/src.valid   \
            -verbose -batch_size 10     \
            -beam_size 1                \
            -seed 1                     \
            -random_sampling_topk -1    \
            -random_sampling_temp 0.0001    \
            -tgt ${DATA_DIR}/morph/tgt.valid   \
            -out $TMP_OUT_DIR/trans_sampling  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/morph/tgt.valid $TMP_OUT_DIR/trans_sampling
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/trans_sampling

echo -n "  [+] Testing LM generation..."
head ${DATA_DIR}/src-test.txt > $TMP_OUT_DIR/src-test.txt
${PYTHON} translate.py -model ${TEST_DIR}/test_model_lm.pt -src $TMP_OUT_DIR/src-test.txt -verbose >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/src-test.txt

echo -n "  [+] Testing LM generation w/ Beam search..."
${PYTHON} translate.py -model ${TEST_DIR}/test_model_lm.pt  \
            -src ${DATA_DIR}/data_lm/src-gen.txt   \
            -verbose -batch_size 10     \
            -beam_size 10 \
            -ban_unk_token \
            -out $TMP_OUT_DIR/gen_beam  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/data_lm/gen-beam-sol.txt $TMP_OUT_DIR/gen_beam
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/gen_beam

echo -n "  [+] Testing LM generation w/ Random Sampling..."
${PYTHON} translate.py -model ${TEST_DIR}/test_model_lm.pt  \
            -src ${DATA_DIR}/data_lm/src-gen.txt   \
            -verbose -batch_size 10     \
            -beam_size 1                \
            -seed 1                     \
            -random_sampling_topk -1    \
            -random_sampling_temp 0.0001    \
            -ban_unk_token \
            -out $TMP_OUT_DIR/gen_sampling  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/data_lm/gen-sampling-sol.txt $TMP_OUT_DIR/gen_sampling
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/gen_sampling

echo -n "  [+] Testing LM generation w/ Random Top-k/Nucleus Sampling..."
${PYTHON} translate.py -model ${TEST_DIR}/test_model_lm.pt  \
            -src ${DATA_DIR}/data_lm/src-gen.txt   \
            -verbose -batch_size 10     \
            -beam_size 1                \
            -seed 3                     \
            -random_sampling_topk -1    \
            -random_sampling_topp 0.95    \
            -random_sampling_temp 1    \
            -ban_unk_token \
            -out $TMP_OUT_DIR/gen_sampling  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/data_lm/gen-nucleus-sampling-sol.txt $TMP_OUT_DIR/gen_sampling
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/gen_sampling

echo -n "  [+] Testing LM generation w/ Random Top-k/Nucleus Sampling and multi beams..."
${PYTHON} translate.py -model ${TEST_DIR}/test_model_lm.pt  \
            -src ${DATA_DIR}/data_lm/src-gen.txt   \
            -verbose -batch_size 10     \
            -beam_size 10                \
            -seed 1                     \
            -random_sampling_topk 50    \
            -random_sampling_topp 0.95    \
            -random_sampling_temp 1    \
            -length_penalty avg \
            -ban_unk_token \
            -min_length 5 \
            -out $TMP_OUT_DIR/gen_sampling  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/data_lm/gen-sampling-beams-sol.txt $TMP_OUT_DIR/gen_sampling
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/gen_sampling

#
# Tools test
#
echo "[+] Doing tools test..."
echo -n "  [+] Doing extract vocabulary test..."
PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH} ${PYTHON} ./tools/extract_vocabulary.py \
            -file $TMP_OUT_DIR/onmt.train.check.vocab.pt -file_type field -side src \
            -out_file $TMP_OUT_DIR/vocab.txt >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
if ! wc -l $TMP_OUT_DIR/vocab.txt | grep -qF  "1002"; then
    echo -n "wrong word count\n" >> ${LOG_FILE}
    wc -l $TMP_OUT_DIR/vocab.txt >> ${LOG_FILE}
    error_exit
fi
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/vocab.txt

echo -n "  [+] Doing embeddings to torch test..."
PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH} ${PYTHON} ./tools/embeddings_to_torch.py \
        -emb_file_enc ${TEST_DIR}/sample_glove.txt \
        -emb_file_dec ${TEST_DIR}/sample_glove.txt \
        -dict_file $TMP_OUT_DIR/onmt.train.check.vocab.pt \
        -output_file $TMP_OUT_DIR/q_gloveembeddings        >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/q_gloveembeddings*

echo -n "  [+] Doing extract embeddings test..."
PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH} ${PYTHON} tools/extract_embeddings.py \
        -model onmt/tests/test_model.pt  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

# Finally, clean up
clean_up
