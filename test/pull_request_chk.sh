#!/bin/bash
# Run this script and fix *any* error before sending PR.

LOG_FILE=/tmp/$$_pull_request_chk.log
echo > ${LOG_FILE} # Empty the log file.

PROJECT_ROOT=`dirname "$0"`"/.."
DATA_DIR="$PROJECT_ROOT/data"
TEST_DIR="$PROJECT_ROOT/test"

clean_up()
{
    rm ${LOG_FILE}
}
trap clean_up SIGINT SIGQUIT SIGKILL

error_exit()
{
    echo "Failed !" | tee -a ${LOG_FILE}
    echo "[!] Check ${LOG_FILE} for detail."
    exit 1
}


# flake8 check
echo -n "[+] Doing flake8 check..."
python -m flake8  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


# unittest
echo -n "[+] Doing unittest test..."
python -m unittest discover >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


# Preprocess test
echo -n "[+] Doing preprocess test..."
python preprocess.py -train_src ${DATA_DIR}/src-train.txt \
		     -train_tgt ${DATA_DIR}/tgt-train.txt \
		     -valid_src ${DATA_DIR}/src-val.txt \
		     -valid_tgt ${DATA_DIR}/tgt-val.txt \
		     -save_data /tmp/data \
		     -src_vocab_size 1000 \
		     -tgt_vocab_size 1000  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


# Translation test
echo -n "[+] Doing translation test..."
head ${DATA_DIR}/src-test.txt > /tmp/src-test.txt
python translate.py -model ${TEST_DIR}/test_model.pt -src /tmp/src-test.txt -verbose >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}


# Train + Translation test
echo -n "[+] Doing preprocess + train + translation test..."
head ${DATA_DIR}/src-val.txt > /tmp/src-val.txt
head ${DATA_DIR}/tgt-val.txt > /tmp/tgt-val.txt
python preprocess.py -train_src /tmp/src-val.txt \
		     -train_tgt /tmp/tgt-val.txt \
		     -valid_src /tmp/src-val.txt \
		     -valid_tgt /tmp/tgt-val.txt \
		     -save_data /tmp/q           \
		     -src_vocab_size 1000        \
		     -tgt_vocab_size 1000        >> ${LOG_FILE} 2>&1
python train.py -data /tmp/q -rnn_size 2 -batch_size 10 \
		-word_vec_size 5 -report_every 5        \
		-rnn_size 10 -epochs 1                 >> ${LOG_FILE} 2>&1
python translate.py -model ${TEST_DIR}/test_model2.pt  \
		    -src ${DATA_DIR}/morph/src.valid   \
		    -verbose -batch_size 10     \
		    -beam_size 10               \
		    -tgt ${DATA_DIR}/morph/tgt.valid   \
		    -out /tmp/trans             >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/morph/tgt.valid /tmp/trans
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

clean_up
