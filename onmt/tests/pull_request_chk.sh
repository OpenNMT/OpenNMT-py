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
        rm -r $TMP_OUT_DIR/dump_pred
        rm -f $TMP_OUT_DIR/*.pt
        rm -rf $TMP_OUT_DIR/sample
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

# black check
echo -n "[+] Doing Black check..."
${PYTHON} -m black --check . >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

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



# Get Vocabulary test

echo -n "[+] Testing vocabulary building..."
PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH} ${PYTHON} onmt/bin/build_vocab.py \
            -config ${DATA_DIR}/data.yaml \
            -save_data $TMP_OUT_DIR/onmt \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.tgt \
            -n_sample 5000 -overwrite >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -f -r $TMP_OUT_DIR/sample

echo -n "[+] Testing vocabulary building with features..."
PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH} ${PYTHON} onmt/bin/build_vocab.py \
            -config ${DATA_DIR}/features_data.yaml \
            -save_data $TMP_OUT_DIR/onmt_feat \
            -src_vocab $TMP_OUT_DIR/onmt_feat.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt_feat.vocab.tgt \
            -n_sample -1  -overwrite>> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -f -r $TMP_OUT_DIR/sample

#
# Training test
#
echo -n "[+] Testing NMT vocab? /transforms prepare..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/data.yaml \
            -save_data $TMP_OUT_DIR/onmt.train.check \
            -dump_fields -dump_transforms -n_sample 30 \
            -overwrite \
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
            -batch_size 10 \
            -num_workers 0 -bucket_size 1024 \
            -word_vec_size 5 -report_every 5 \
            -hidden_size 10 -train_steps 10 \
            -tensorboard "true" \
            -tensorboard_log_dir $TMP_OUT_DIR/logs_train >> ${LOG_FILE} 2>&1
${PYTHON} onmt/tests/test_events.py --logdir $TMP_OUT_DIR/logs_train -tensorboard_checks train
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -r $TMP_OUT_DIR/logs_train

echo -n "  [+] Testing NMT training and validation w/ copy..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -batch_size 10 \
            -num_workers 0 -bucket_size 1024 \
            -word_vec_size 5 -report_every 2 \
            -hidden_size 10 -train_steps 10 -valid_steps 5 \
            -tensorboard "true" \
            -tensorboard_log_dir $TMP_OUT_DIR/logs_train_and_valid \
            -copy_attn >> ${LOG_FILE} 2>&1
${PYTHON} onmt/tests/test_events.py --logdir $TMP_OUT_DIR/logs_train_and_valid -tensorboard_checks train
${PYTHON} onmt/tests/test_events.py --logdir $TMP_OUT_DIR/logs_train_and_valid -tensorboard_checks valid
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -r $TMP_OUT_DIR/logs_train_and_valid

echo -n "  [+] Testing NMT training w/ align..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/align_data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -num_workers 0 -bucket_size 1024 \
            -encoder_type transformer -decoder_type transformer \
            -layers 4 -word_vec_size 16 -hidden_size 16 -heads 2 -transformer_ff 64 \
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
            -batch_size 10 \
            -num_workers 0 -bucket_size 1024 \
            -word_vec_size 5 -report_every 5        \
            -coverage_attn true -lambda_coverage 0.1 \
            -hidden_size 10 -train_steps 10 >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing NMT transformer training w/ validation with dynamic scoring and copy ..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -encoder_type transformer \
            -decoder_type transformer \
            -layers 4 \
            -word_vec_size 16 \
            -hidden_size 16 \
            -num_workers 0 -bucket_size 1024 \
            -heads 2 \
            -transformer_ff 64 \
            -bucket_size 1024 \
            -train_steps 10 \
            -report_every 2 \
            -valid_steps 5 \
            -valid_metrics "BLEU" "TER" \
            -tensorboard "true" \
            -scoring_debug "true" \
            -copy_attn \
            -position_encoding \
            -dump_preds $TMP_OUT_DIR/dump_pred \
            -tensorboard_log_dir $TMP_OUT_DIR/logs_dynamic-scoring_and_copy >> ${LOG_FILE} 2>&1
      
${PYTHON} onmt/tests/test_events.py --logdir $TMP_OUT_DIR/logs_dynamic-scoring_and_copy -tensorboard_checks valid_metrics
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -r $TMP_OUT_DIR/logs_dynamic-scoring_and_copy

echo -n "  [+] Testing NMT transformer training w/ validation with dynamic scoring and maxrelative ..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -encoder_type transformer \
            -decoder_type transformer \
            -layers 4 \
            -word_vec_size 16 \
            -hidden_size 16 \
            -num_workers 0 -bucket_size 1024 \
            -heads 2 \
            -transformer_ff 64 \
            -bucket_size 1024 \
            -train_steps 10 \
            -report_every 2 \
            -valid_steps 5 \
            -valid_metrics "BLEU" "TER" \
            -tensorboard "true" \
            -scoring_debug "true" \
            -max_relative_positions 8 \
            -dump_preds $TMP_OUT_DIR/dump_pred \
            -tensorboard_log_dir $TMP_OUT_DIR/logs_dynamic-scoring_and_relative >> ${LOG_FILE} 2>&1
      
${PYTHON} onmt/tests/test_events.py --logdir $TMP_OUT_DIR/logs_dynamic-scoring_and_relative -tensorboard_checks valid_metrics
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -r $TMP_OUT_DIR/logs_dynamic-scoring_and_relative

echo -n "  [+] Testing NMT transformer training w/ validation with dynamic scoring and rotary ..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -encoder_type transformer \
            -decoder_type transformer \
            -layers 4 \
            -word_vec_size 16 \
            -hidden_size 16 \
            -num_workers 0 -bucket_size 1024 \
            -heads 2 \
            -transformer_ff 64 \
            -bucket_size 1024 \
            -train_steps 10 \
            -report_every 2 \
            -valid_steps 5 \
            -valid_metrics "BLEU" "TER" \
            -tensorboard "true" \
            -scoring_debug "true" \
            -max_relative_positions -1 \
            -dump_preds $TMP_OUT_DIR/dump_pred \
            -tensorboard_log_dir $TMP_OUT_DIR/logs_dynamic-scoring_and_rotary >> ${LOG_FILE} 2>&1
      
${PYTHON} onmt/tests/test_events.py --logdir $TMP_OUT_DIR/logs_dynamic-scoring_and_rotary -tensorboard_checks valid_metrics
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -r $TMP_OUT_DIR/logs_dynamic-scoring_and_rotary

echo -n "  [+] Testing NMT transformer training w/ validation with dynamic scoring and alibi ..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -encoder_type transformer \
            -decoder_type transformer \
            -layers 4 \
            -word_vec_size 16 \
            -hidden_size 16 \
            -num_workers 0 -bucket_size 1024 \
            -heads 2 \
            -transformer_ff 64 \
            -bucket_size 1024 \
            -train_steps 10 \
            -report_every 2 \
            -valid_steps 5 \
            -valid_metrics "BLEU" "TER" \
            -tensorboard "true" \
            -scoring_debug "true" \
            -max_relative_positions -2 \
            -dump_preds $TMP_OUT_DIR/dump_pred \
            -tensorboard_log_dir $TMP_OUT_DIR/logs_dynamic-scoring_and_alibi >> ${LOG_FILE} 2>&1
      
${PYTHON} onmt/tests/test_events.py --logdir $TMP_OUT_DIR/logs_dynamic-scoring_and_alibi -tensorboard_checks valid_metrics
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -r $TMP_OUT_DIR/logs_dynamic-scoring_and_alibi

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
            -num_workers 0 -bucket_size 1024 \
            -dec_layers 2 -batch_size 10 \
            -heads 4 -transformer_ff 64 \
            -word_vec_size 16 -report_every 5        \
            -hidden_size 16 -train_steps 10 >> ${LOG_FILE} 2>&1
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
            -num_workers 0 -bucket_size 1024 \
            -word_vec_size 16 -report_every 5        \
            -hidden_size 16 -train_steps 10 \
            -copy_attn >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}*

echo -n "  [+] Testing Checkpoint Vocabulary Update..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.tgt \
            -src_vocab_size 1000 -tgt_vocab_size 1000 \
            -batch_size 10 \
            -word_vec_size 5 -hidden_size 10 \
            -num_workers 0 -bucket_size 1024 \
            -report_every 5 -train_steps 10 \
            -save_model $TMP_OUT_DIR/onmt.model \
            -save_checkpoint_steps 10 >> ${LOG_FILE} 2>&1
sed -i '1s/^/new_tok\t100000000\n/' $TMP_OUT_DIR/onmt.vocab.src >> ${LOG_FILE} 2>&1
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt.vocab.tgt \
            -src_vocab_size 1000 -tgt_vocab_size 1000 \
            -batch_size 10 \
            -word_vec_size 5 -hidden_size 10 \
            -num_workers 0 -bucket_size 1024 \
            -report_every 5 -train_steps 20 \
            -update_vocab -reset_optim "states" \
            -train_from $TMP_OUT_DIR/onmt.model_step_10.pt >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing Checkpoint Vocabulary Update with LM..."
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
            -num_workers 0 -bucket_size 1024 \
            -word_vec_size 16 -report_every 5 \
            -save_model $TMP_OUT_DIR/lm.onmt.model \
            -save_checkpoint_steps 10 \
            -hidden_size 16 -train_steps 10 >> ${LOG_FILE} 2>&1
sed -i '1s/^/new_tok2\t100000000\n/' $TMP_OUT_DIR/onmt.vocab.src >> ${LOG_FILE} 2>&1
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
            -num_workers 0 -bucket_size 1024 \
            -heads 4 -transformer_ff 64 \
            -word_vec_size 16 -report_every 5 \
            -hidden_size 16  -train_steps 20 \
            -update_vocab -reset_optim "states" \
            -train_from $TMP_OUT_DIR/lm.onmt.model_step_10.pt >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing Graph Neural Network training..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/ggnn_data.yaml \
            -src_seq_length 1000 -tgt_seq_length 30 \
            -encoder_type ggnn -layers 2 \
            -decoder_type rnn -hidden_size 256 \
            -learning_rate 0.1 -learning_rate_decay 0.8 \
            -num_workers 0 -bucket_size 1024 \
            -global_attention general -batch_size 32 -word_vec_size 256 \
            -bridge -train_steps 10 -n_edge_types 9 -state_dim 256 \
            -n_steps 10 -n_node 64 >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing training with features..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/features_data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt_feat.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt_feat.vocab.tgt \
            -src_vocab_size 1000 -tgt_vocab_size 1000 \
            -batch_size 10 \
            -word_vec_size 5 -hidden_size 10 \
            -num_workers 0 -bucket_size 1024 \
            -report_every 5 -train_steps 10 \
            -save_model $TMP_OUT_DIR/onmt.features.model \
            -save_checkpoint_steps 10 >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


echo -n "  [+] Testing training with features and dynamic scoring..."
${PYTHON} onmt/bin/train.py \
            -config ${DATA_DIR}/features_data.yaml \
            -src_vocab $TMP_OUT_DIR/onmt_feat.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/onmt_feat.vocab.tgt \
            -src_vocab_size 1000 -tgt_vocab_size 1000 \
            -batch_size 10 \
            -word_vec_size 5 -hidden_size 10 \
            -num_workers 0 -bucket_size 1024 \
            -report_every 5 -train_steps 10 -valid_steps 5\
            -valid_metrics "BLEU" "TER" \
            -save_model $TMP_OUT_DIR/onmt.features.model \
            -save_checkpoint_steps 10 >> ${LOG_FILE} 2>&1

[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -f $TMP_OUT_DIR/onmt.vocab*
rm -f $TMP_OUT_DIR/onmt.model*
rm -f $TMP_OUT_DIR/onmt_feat.vocab.*

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

echo -n "  [+] Testing NMT translation with features..."
${PYTHON} translate.py \
            -model ${TMP_OUT_DIR}/onmt.features.model_step_10.pt \
            -src ${DATA_DIR}/data_features/src-test-with-feats.txt \
            -n_src_feats 1 -verbose >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -f $TMP_OUT_DIR/onmt.features.model*

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
echo "  [+] Testing LM generation..." | tee -a ${LOG_FILE}
head ${DATA_DIR}/src-test.txt > $TMP_OUT_DIR/src-test.txt
${PYTHON} translate.py -model ${TEST_DIR}/test_model_lm.pt -src $TMP_OUT_DIR/src-test.txt -verbose >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/src-test.txt

echo -n "  [+] Testing LM generation w/ Beam search..."
echo "  [+] Testing LM generation w/ Beam search..." | tee -a ${LOG_FILE}
${PYTHON} translate.py -model ${TEST_DIR}/test_model_lm.pt  \
            -src ${DATA_DIR}/data_lm/src-gen.txt   \
            -verbose -batch_size 1     \
            -beam_size 10 \
            -ban_unk_token \
            -length_penalty none \
            -out $TMP_OUT_DIR/gen_beam  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/data_lm/gen-beam-sol.txt $TMP_OUT_DIR/gen_beam
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/gen_beam

echo -n "  [+] Testing LM generation w/ Random Sampling..."
${PYTHON} translate.py -model ${TEST_DIR}/test_model_lm.pt  \
            -src ${DATA_DIR}/data_lm/src-gen.txt   \
            -verbose -batch_size 1     \
            -beam_size 1                \
            -seed 1                     \
            -random_sampling_topk -1    \
            -random_sampling_temp 0.0001    \
            -ban_unk_token \
            -length_penalty none \
            -out $TMP_OUT_DIR/gen_sampling  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/data_lm/gen-sampling-sol.txt $TMP_OUT_DIR/gen_sampling
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/gen_sampling

echo -n "  [+] Testing LM generation w/ Random Top-k/Nucleus Sampling..."
${PYTHON} translate.py -model ${TEST_DIR}/test_model_lm.pt  \
            -src ${DATA_DIR}/data_lm/src-gen.txt   \
            -verbose -batch_size 1     \
            -beam_size 1                \
            -seed 3                     \
            -random_sampling_topk -1    \
            -random_sampling_topp 0.95    \
            -random_sampling_temp 1    \
            -ban_unk_token \
            -length_penalty none \
            -out $TMP_OUT_DIR/gen_sampling  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/data_lm/gen-nucleus-sampling-sol$(${PYTHON}  -c "import torch; print(torch.__version__[0])").txt $TMP_OUT_DIR/gen_sampling
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/gen_sampling

echo -n "  [+] Testing LM generation w/ Random Top-k/Nucleus Sampling and multi beams..."
${PYTHON} translate.py -model ${TEST_DIR}/test_model_lm.pt  \
            -src ${DATA_DIR}/data_lm/src-gen.txt   \
            -verbose -batch_size 1     \
            -beam_size 10                \
            -seed 2                     \
            -random_sampling_topk 50    \
            -random_sampling_topp 0.95    \
            -random_sampling_temp 1    \
            -length_penalty avg \
            -ban_unk_token \
            -min_length 5 \
            -out $TMP_OUT_DIR/gen_sampling  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/data_lm/gen-sampling-beams-sol$($PYTHON -c "import torch; print(torch.__version__[0])").txt $TMP_OUT_DIR/gen_sampling
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/gen_sampling

#
# Inference engines test
#
echo -n "  [+] Testing PY LM inference engine .."
echo "  [+] Testing PY LM inference engine .."| tee -a ${LOG_FILE}
head ${DATA_DIR}/src-test.txt > $TMP_OUT_DIR/src-test.txt
${PYTHON} onmt/tests/test_inference_engines.py -model ${TEST_DIR}/test_model_lm.pt \
            -model_task lm \
            -input_file $TMP_OUT_DIR/src-test.txt \
            -inference_config_file ${DATA_DIR}/inference-engine_py.yaml \
            -inference_mode py \
            -out $TMP_OUT_DIR/inference_engine_lm_py_outputs  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/src-test.txt
rm $TMP_OUT_DIR/inference_engine_lm_py_outputs_file.json
rm $TMP_OUT_DIR/inference_engine_lm_py_outputs_list.json

echo "  [+] Testing CT2 LM inference engine .."| tee -a ${LOG_FILE}
head ${DATA_DIR}/src-test.txt > $TMP_OUT_DIR/src-test.txt
${PYTHON} onmt/tests/test_inference_engines.py -model ${TEST_DIR}/test_model_lm_ct2 \
            -model_task lm \
            -input_file $TMP_OUT_DIR/src-test.txt \
            -inference_config_file ${DATA_DIR}/inference-engine_py.yaml \
            -inference_mode ct2 \
            -out $TMP_OUT_DIR/inference_engine_lm_ct2_outputs  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/src-test.txt
rm $TMP_OUT_DIR/inference_engine_lm_ct2_outputs_file.json
rm $TMP_OUT_DIR/inference_engine_lm_ct2_outputs_list.json

echo -n "  [+] Testing PY SEQ2SEQ inference engine .."
echo "  [+] Testing PY SEQ2SEQ inference engine .."| tee -a ${LOG_FILE}
head ${DATA_DIR}/src-test.txt > $TMP_OUT_DIR/src-test.txt
${PYTHON} onmt/tests/test_inference_engines.py -model ${TEST_DIR}/test_model.pt \
            -model_task seq2seq \
            -input_file $TMP_OUT_DIR/src-test.txt \
            -inference_config_file ${DATA_DIR}/inference-engine_py.yaml \
            -inference_mode py \
            -out $TMP_OUT_DIR/inference_engine_seq2seq_py_outputs  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/src-test.txt
rm $TMP_OUT_DIR/inference_engine_seq2seq_py_outputs_file.json
rm $TMP_OUT_DIR/inference_engine_seq2seq_py_outputs_list.json

#
# Tools test
#
echo "[+] Doing tools test..."
echo -n "  [+] Doing extract vocabulary test..."
PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH} ${PYTHON} ./tools/extract_vocabulary.py \
            -model ${TEST_DIR}/test_model.pt -side src \
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
        -dict_file ${TEST_DIR}/test_model.pt \
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
