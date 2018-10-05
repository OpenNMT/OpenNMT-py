#!/bin/bash
# Run this script and fix *any* error before sending PR.

LOG_FILE=/tmp/$$_pull_request_chk.log
echo > ${LOG_FILE} # Empty the log file.

PROJECT_ROOT=`dirname "$0"`"/../../"
DATA_DIR="$PROJECT_ROOT/data"
TEST_DIR="$PROJECT_ROOT/onmt/tests"
PYTHON="python"

clean_up()
{
    rm ${LOG_FILE}
    rm -rf /tmp/*pt
    rm -rf /tmp/im2text
    rm -rf /tm/speech
}
trap clean_up SIGINT SIGQUIT SIGKILL

error_exit()
{
    echo "Failed !" | tee -a ${LOG_FILE}
    echo "[!] Check ${LOG_FILE} for detail."
    exit 1
}

environment_prepare()
{
  # Download img2text corpus
  wget -q -O /tmp/im2text.tgz http://lstm.seas.harvard.edu/latex/im2text_small.tgz
  tar zxf /tmp/im2text.tgz -C /tmp/
  head /tmp/im2text/src-train.txt > /tmp/im2text/src-train-head.txt
  head /tmp/im2text/tgt-train.txt > /tmp/im2text/tgt-train-head.txt
  head /tmp/im2text/src-val.txt > /tmp/im2text/src-val-head.txt
  head /tmp/im2text/tgt-val.txt > /tmp/im2text/tgt-val-head.txt

  wget -q -O /tmp/test_model_speech.pt http://lstm.seas.harvard.edu/latex/model_step_2760.pt

  # Download speech2text corpus
  wget -q -O /tmp/speech.tgz http://lstm.seas.harvard.edu/latex/speech.tgz
  tar zxf /tmp/speech.tgz -C /tmp/
  head /tmp/speech/src-train.txt > /tmp/speech/src-train-head.txt
  head /tmp/speech/tgt-train.txt > /tmp/speech/tgt-train-head.txt
  head /tmp/speech/src-val.txt > /tmp/speech/src-val-head.txt
  head /tmp/speech/tgt-val.txt > /tmp/speech/tgt-val-head.txt

  wget -q -O /tmp/test_model_im2text.pt http://lstm.seas.harvard.edu/latex/test_model_im2text.pt
}

# flake8 check
echo -n "[+] Doing flake8 check..."
${PYTHON} -m flake8 >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


# Environment prepartion
echo -n "[+] Preparing for test..."
environment_prepare
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


# unittest
echo -n "[+] Doing unittest test..."
${PYTHON} -m unittest discover >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


#
# Preprocess test
#
echo "[+] Doing preprocess test..."

echo -n "  [+] Testing NMT preprocessing..."
rm -rf /tmp/data*pt
${PYTHON} preprocess.py -train_src ${DATA_DIR}/src-train.txt \
		     -train_tgt ${DATA_DIR}/tgt-train.txt \
		     -valid_src ${DATA_DIR}/src-val.txt \
		     -valid_tgt ${DATA_DIR}/tgt-val.txt \
		     -save_data /tmp/data \
		     -src_vocab_size 1000 \
		     -tgt_vocab_size 1000  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing img2text preprocessing..."
rm -rf /tmp/im2text/data*pt
${PYTHON} preprocess.py -data_type img \
		     -src_dir /tmp/im2text/images \
		     -train_src /tmp/im2text/src-train.txt \
		     -train_tgt /tmp/im2text/tgt-train.txt \
		     -valid_src /tmp/im2text/src-val.txt \
		     -valid_tgt /tmp/im2text/tgt-val.txt \
		     -save_data /tmp/im2text/data  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing speech2text preprocessing..."
rm -rf /tmp/speech/data*pt
${PYTHON} preprocess.py -data_type audio \
		     -src_dir /tmp/speech/an4_dataset \
		     -train_src /tmp/speech/src-train.txt \
		     -train_tgt /tmp/speech/tgt-train.txt \
		     -valid_src /tmp/speech/src-val.txt \
		     -valid_tgt /tmp/speech/tgt-val.txt \
		     -save_data /tmp/speech/data  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


#
# Translation test
#
echo "[+] Doing translation test..."

echo -n "  [+] Testing NMT translation..."
head ${DATA_DIR}/src-test.txt > /tmp/src-test.txt
${PYTHON} translate.py -model ${TEST_DIR}/test_model.pt -src /tmp/src-test.txt -verbose >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing img2text translation..."
head /tmp/im2text/src-val.txt > /tmp/im2text/src-val-head.txt
head /tmp/im2text/tgt-val.txt > /tmp/im2text/tgt-val-head.txt
${PYTHON} translate.py -data_type img \
	            -src_dir /tmp/im2text/images \
		    -model /tmp/test_model_im2text.pt \
		    -src /tmp/im2text/src-val-head.txt \
		    -tgt /tmp/im2text/tgt-val-head.txt \
		    -verbose -out /tmp/im2text/trans  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing speech2text translation..."
head /tmp/speech/src-val.txt > /tmp/speech/src-val-head.txt
head /tmp/speech/tgt-val.txt > /tmp/speech/tgt-val-head.txt
${PYTHON} translate.py -data_type audio \
	            -src_dir /tmp/speech/an4_dataset \
		    -model /tmp/test_model_speech.pt \
		    -src /tmp/speech/src-val-head.txt \
		    -tgt /tmp/speech/tgt-val-head.txt \
		    -verbose -out /tmp/speech/trans  >> ${LOG_FILE} 2>&1
diff /tmp/speech/tgt-val-head.txt /tmp/speech/trans
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


# NMT Preprocess + Train + Translation test
echo -n "[+] Doing NMT {preprocess + train + translation} test..."
head ${DATA_DIR}/src-val.txt > /tmp/src-val.txt
head ${DATA_DIR}/tgt-val.txt > /tmp/tgt-val.txt
rm -rf /tmp/q*pt
${PYTHON} preprocess.py -train_src /tmp/src-val.txt \
		     -train_tgt /tmp/tgt-val.txt \
		     -valid_src /tmp/src-val.txt \
		     -valid_tgt /tmp/tgt-val.txt \
		     -save_data /tmp/q           \
		     -src_vocab_size 1000        \
		     -tgt_vocab_size 1000        >> ${LOG_FILE} 2>&1
${PYTHON} train.py -data /tmp/q -rnn_size 2 -batch_size 10 \
		-word_vec_size 5 -report_every 5        \
		-rnn_size 10 -train_steps 10        >> ${LOG_FILE} 2>&1
${PYTHON} translate.py -model ${TEST_DIR}/test_model2.pt  \
		    -src ${DATA_DIR}/morph/src.valid   \
		    -verbose -batch_size 10     \
		    -beam_size 10               \
		    -tgt ${DATA_DIR}/morph/tgt.valid   \
		    -out /tmp/trans             >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/morph/tgt.valid /tmp/trans
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


# NMT Preprocess w/sharding + Train w/copy
echo -n "[+] Doing NMT {preprocess w/sharding + train w/copy} test..."
head ${DATA_DIR}/src-val.txt > /tmp/src-val.txt
head ${DATA_DIR}/tgt-val.txt > /tmp/tgt-val.txt
rm -rf /tmp/q*pt
${PYTHON} preprocess.py -train_src /tmp/src-val.txt \
		     -train_tgt /tmp/tgt-val.txt \
		     -valid_src /tmp/src-val.txt \
		     -valid_tgt /tmp/tgt-val.txt \
		     -save_data /tmp/q           \
		     -src_vocab_size 1000        \
		     -tgt_vocab_size 1000        \
		     -max_shard_size 1           \
             -dynamic_dict               >> ${LOG_FILE} 2>&1
${PYTHON} train.py -data /tmp/q -rnn_size 2 -batch_size 10 \
		-word_vec_size 5 -report_every 5        \
		-rnn_size 10 -train_steps 10 -copy_attn       >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


echo -n "[+] Doing im2text {preprocess w/sharding + train} test..."
head /tmp/im2text/src-val.txt > /tmp/im2text/src-val-head.txt
head /tmp/im2text/tgt-val.txt > /tmp/im2text/tgt-val-head.txt
rm -rf /tmp/im2text/q*pt
${PYTHON} preprocess.py -data_type img \
	             -src_dir /tmp/im2text/images \
		     -train_src /tmp/im2text/src-val-head.txt \
		     -train_tgt /tmp/im2text/tgt-val-head.txt \
		     -valid_src /tmp/im2text/src-val-head.txt \
		     -valid_tgt /tmp/im2text/tgt-val-head.txt \
             -shard_size 5 \
		     -save_data /tmp/im2text/q  >> ${LOG_FILE} 2>&1
${PYTHON} train.py -model_type img \
	        -data /tmp/im2text/q -rnn_size 2 -batch_size 10 \
		-word_vec_size 5 -report_every 5 -rnn_size 10 -train_steps 10  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


echo -n "[+] Doing speech2text {preprocess + train} test..."
head /tmp/speech/src-val.txt > /tmp/speech/src-val-head.txt
head /tmp/speech/tgt-val.txt > /tmp/speech/tgt-val-head.txt
rm -rf /tmp/speech/q*pt
${PYTHON} preprocess.py -data_type audio \
	             -src_dir /tmp/speech/an4_dataset \
		     -train_src /tmp/speech/src-val-head.txt \
		     -train_tgt /tmp/speech/tgt-val-head.txt \
		     -valid_src /tmp/speech/src-val-head.txt \
		     -valid_tgt /tmp/speech/tgt-val-head.txt \
             -shard_size 50 \
		     -save_data /tmp/speech/q  >> ${LOG_FILE} 2>&1
${PYTHON} train.py -model_type audio \
	        -data /tmp/speech/q -rnn_size 2 -batch_size 10 \
		-word_vec_size 5 -report_every 5 -rnn_size 10 -train_steps 10  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


# Finally, clean up
clean_up
