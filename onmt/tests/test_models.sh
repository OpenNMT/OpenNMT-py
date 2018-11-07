#!/bin/bash
### Utility to test models by using command lines actions to run
### Actions are typically model training / translation
### or options setter.
###
### Actions are executed in the order its provided, therefore setters must
### be first
###  
### Example: 
###     - Run all tests: 
###             ./test_models.sh all
###             or
###             ./test_models.sh
###
###     - Run all tests using GPU (i.e. -gpuid 0):
###             ./test_models.sh set_gpu all
###             (note that set_gpu comes first!!!)
###             you can set all GPU (i.e. to match CUDA_VISIBLE_DEVICES)!
###             ./test_models.sh set_all_gpu all
###  
###     - Train each models, and run translation (for each!):
###             ./test_models.sh translate_each all
###             (note that translate_each comes first!!!)
###     
###     - Train and translate a specific model (e.g. lstm):
###             ./test_models.sh lstm translate
###             note that translate only consider the last model therefore:
###             ./test_models.sh lstm cnn translate
###             would actually use CNN model for translation
###  
###     - Run in debug mode (stops on first error) 
###             ./test_models set_debug all 
###  


PYTHON_BIN=python


MODEL_DIR="/tmp"
MODEL_NAME="onmt_tmp_model"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
MODEL_FILES_PREFIX="${MODEL_NAME}_acc_"

TEST_DIR="./onmt/tests"
TEST_MODEL_NAME="test_model.pt"
TEST_MODEL_PATH="$TEST_DIR/$TEST_MODEL_NAME"

DATA_DIR="./data"
DATA_PATH="$DATA_DIR/data"


# Do not edit directly, use calls 'set_gpu' and 'translate_each'
GPUID=-1
TRANSLATE_EACH=0

### Some setters
###############################################
set_gpu(){
    GPUID=0
}


set_all_gpu(){
    GPUID=$(sed 's/,/ /g' <(echo $CUDA_VISIBLE_DEVICES) >&1)
}
print_gpuid(){
    echo "$GPUID"
}

set_debug(){
    set -e
}

translate_each(){
    TRANSLATE_EACH=1
}
### Some utils functions
###############################################
mv_best_checkpoint(){
    best_model="$(ls -lsrt $MODEL_DIR | grep "${MODEL_FILES_PREFIX}*" | tail -n 1 | awk '{print $NF}')"
    mv "$MODEL_DIR/$best_model" "$TEST_MODEL_PATH"
}

rm_tmp_checkpoints(){
    rm -f "$MODEL_DIR/${MODEL_FILES_PREFIX}"*
}


### RNNLM
###############################################
lstm(){
    rm -f "$DATA_DIR"/*.pt
    $PYTHON_BIN preprocess.py -train_src "$DATA_DIR"/src-train.txt \
                         -train_tgt "$DATA_DIR"/tgt-train.txt \
                         -valid_src "$DATA_DIR"/src-val.txt \
                         -valid_tgt "$DATA_DIR"/tgt-val.txt \
                         -save_data "$DATA_PATH" \
                         -src_vocab_size 1000 \
                         -tgt_vocab_size 1000

    $PYTHON_BIN train.py -data "$DATA_PATH" \
                    -save_model "$MODEL_PATH" \
                    -gpuid $GPUID \
                    -rnn_size 512 \
                    -word_vec_size 512 \
                    -layers 1 \
                    -train_steps 10000 \
                    -optim adam  \
                    -learning_rate 0.001 \
                    -rnn_type LSTM
    mv_best_checkpoint
    maybe_translate
    rm_tmp_checkpoints
}



### SRU
###############################################
sru(){
    rm -f "$DATA_DIR"/*.pt
    $PYTHON_BIN preprocess.py -train_src "$DATA_DIR"/src-train.txt \
                         -train_tgt "$DATA_DIR"/tgt-train.txt \
                         -valid_src "$DATA_DIR"/src-val.txt \
                         -valid_tgt "$DATA_DIR"/tgt-val.txt \
                         -save_data "$DATA_PATH" \
                         -src_vocab_size 1000 \
                         -tgt_vocab_size 1000 \
                         -rnn_type "SRU" \
                         -input_feed 0

    $PYTHON_BIN train.py -data "$DATA_PATH" \
                    -save_model "$MODEL_PATH" \
                    -gpuid $GPUID \
                    -rnn_size 512 \
                    -word_vec_size 512 \
                    -layers 1 \
                    -train_steps 10000 \
                    -optim adam  \
                    -learning_rate 0.001 \
                    -rnn_type LSTM
    mv_best_checkpoint
    maybe_translate
    rm_tmp_checkpoints
}
### CNN
###############################################
cnn(){
    rm -f "$DATA_DIR"/*.pt
    $PYTHON_BIN preprocess.py -train_src "$DATA_DIR"/src-train.txt\
                         -train_tgt "$DATA_DIR"/tgt-train.txt \
                         -valid_src "$DATA_DIR"/src-val.txt \
                         -valid_tgt "$DATA_DIR"/tgt-val.txt \
                         -save_data "$DATA_PATH" \
                         -src_vocab_size 1000 \
                         -tgt_vocab_size 1000 
    
    $PYTHON_BIN train.py -data "$DATA_PATH" \
                    -save_model "$MODEL_PATH" \
                    -gpuid $GPUID \
                    -rnn_size 256 \
                    -word_vec_size 256 \
                    -layers 2 \
                    -train_steps 10000 \
                    -optim adam  \
                    -learning_rate 0.001 \
                    -encoder_type cnn \
                    -decoder_type cnn
    mv_best_checkpoint
    maybe_translate
    rm_tmp_checkpoints
}


### MORPH DATA
###############################################
morph(){
    ################# MORPH DATA
    rm -f "$DATA_DIR"/morph/*.pt
    $PYTHON_BIN preprocess.py -train_src "$DATA_DIR"/morph/src.train \
                         -train_tgt "$DATA_DIR"/morph/tgt.train \
                         -valid_src "$DATA_DIR"/morph/src.valid \
                         -valid_tgt "$DATA_DIR"/morph/tgt.valid \
                         -save_data "$DATA_DIR"/morph/data 

    $PYTHON_BIN train.py -data "$DATA_DIR"/morph/data \
                    -save_model "$MODEL_PATH" \
                    -gpuid $GPUID \
                    -rnn_size 400 \
                    -word_vec_size 100 \
                    -layers 1 \
                    -train_steps 10000 \
                    -optim adam  \
                    -learning_rate 0.001

    mv_best_checkpoint
    maybe_translate
    rm_tmp_checkpoints
}


### TRANSFORMER
###############################################
transformer(){
    rm -f "$DATA_DIR"/*.pt
    $PYTHON_BIN preprocess.py -train_src "$DATA_DIR"/src-train.txt \
                         -train_tgt "$DATA_DIR"/tgt-train.txt \
                         -valid_src "$DATA_DIR"/src-val.txt \
                         -valid_tgt "$DATA_DIR"/tgt-val.txt \
                         -save_data "$DATA_PATH" \
                         -src_vocab_size 1000 \
                         -tgt_vocab_size 1000 \
                         -share_vocab


    $PYTHON_BIN train.py -data "$DATA_PATH" \
                    -save_model "$MODEL_PATH" \
                    -share_embedding \
                    -batch_type tokens \
                    -batch_size 1024 \
                    -accum_count 4 \
                    -layers 1 \
                    -rnn_size 256 \
                    -word_vec_size 256 \
                    -encoder_type transformer \
                    -decoder_type transformer \
                    -train_steps 10000 \
                    -gpuid $GPUID \
                    -max_generator_batches 4 \
                    -dropout 0.1 \
                    -normalization tokens \
                    -max_grad_norm 0 \
                    -optim adam \
                    -decay_method noam \
                    -learning_rate 2 \
                    -position_encoding \
                    -param_init 0 \
                    -warmup_steps 100 \
                    -param_init_glorot \
                    -adam_beta2 0.998

    mv_best_checkpoint
    maybe_translate
    rm_tmp_checkpoints

}


### TRANSLATION
###############################################
translate(){
    $PYTHON_BIN translate.py -gpu "$GPUID" \
                        -model "$TEST_MODEL_PATH" \
                        -output "$TEST_DIR"/output_hyp.txt \
                        -beam 5 \
                        -batch_size 32 \
                        -src "$DATA_DIR"/src-val.txt
}

maybe_translate(){
    if [ $TRANSLATE_EACH -eq 1 ]
    then
        translate
    fi
}

all(){
    lstm
    sru
    cnn
    morph
    transformer
    translate

}

actions="$@"

# set the default action
if [ -z "$1" ]; then
    actions="all"
fi

# Process actions (in order)
for action in $actions; do
    echo "Running: $action"
    eval "$action"
done

echo "Done."
