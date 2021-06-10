# # Retrain the models used for CI.
# # Should be done rarely, indicates a major breaking change. 
my_python=python

############### TEST regular RNN choose either -rnn_type LSTM / GRU / SRU and set input_feed 0 for SRU
if false; then
$my_python build_vocab.py \
    -config data/data.yaml -save_data data/data \
    -src_vocab data/data.vocab.src -tgt_vocab data/data.vocab.tgt \
    -overwrite true
$my_python train.py \
    -config data/data.yaml -src_vocab data/data.vocab.src -tgt_vocab data/data.vocab.tgt \
    -src_vocab_size 1000 -tgt_vocab_size 1000 \
    -save_model tmp -world_size 1 -gpu_ranks 0 \
    -rnn_type LSTM -input_feed 0 \
    -rnn_size 256 -word_vec_size 256 \
    -layers 1 -train_steps 10000 \
    -optim adam  -learning_rate 0.001
    # -truncated_decoder 5 
    # -label_smoothing 0.1

mv tmp*10000.pt onmt/tests/test_model.pt
rm tmp*.pt
fi


############### TEST CNN 
if false; then
$my_python build_vocab.py \
    -config data/data.yaml -save_data data/data \
    -src_vocab data/data.vocab.src -tgt_vocab data/data.vocab.tgt \
    -overwrite true
$my_python train.py \
    -config data/data.yaml -src_vocab data/data.vocab.src -tgt_vocab data/data.vocab.tgt \
    -src_vocab_size 1000 -tgt_vocab_size 1000 \
    -save_model /tmp/tmp -world_size 1 -gpu_ranks 0 \
    -encoder_type cnn -decoder_type cnn \
    -rnn_size 256 -word_vec_size 256 \
    -layers 2 -train_steps 10000 \
    -optim adam  -learning_rate 0.001

mv /tmp/tmp*10000.pt onmt/tests/test_model.pt
rm /tmp/tmp*.pt
fi

################# MORPH DATA
if false; then
$my_python build_vocab.py \
    -config data/morph_data.yaml -save_data data/data \
    -src_vocab data/morph_data.vocab.src -tgt_vocab data/morph_data.vocab.tgt \
    -overwrite true
$my_python train.py \
    -config data/morph_data.yaml -src_vocab data/morph_data.vocab.src -tgt_vocab data/morph_data.vocab.tgt \
    -save_model tmp -world_size 1 -gpu_ranks 0 \
    -rnn_size 400 -word_vec_size 100 \
    -layers 1 -train_steps 8000 \
    -optim adam  -learning_rate 0.001


mv tmp*8000.pt onmt/tests/test_model2.pt

rm tmp*.pt
fi


############### TEST TRANSFORMER
if false; then
$my_python build_vocab.py \
    -config data/data.yaml -save_data data/data \
    -src_vocab data/data.vocab.src -tgt_vocab data/data.vocab.tgt \
    -overwrite true -share_vocab

$my_python train.py \
    -config data/data.yaml -src_vocab data/data.vocab.src -tgt_vocab data/data.vocab.tgt \
    -save_model /tmp/tmp \
    -batch_type tokens -batch_size 1024 -accum_count 4 \
    -layers 4 -rnn_size 256 -word_vec_size 256 \
    -encoder_type transformer -decoder_type transformer \
    -share_embedding -share_vocab \
    -train_steps 10000 -world_size 1 -gpu_ranks 0 \
    -max_generator_batches 4 -dropout 0.1 \
    -normalization tokens \
    -max_grad_norm 0 -optim adam -decay_method noam \
    -learning_rate 2 -label_smoothing 0.1 \
    -position_encoding -param_init 0 \
    -warmup_steps 100 -param_init_glorot -adam_beta2 0.998

mv /tmp/tmp*10000.pt onmt/tests/test_model.pt
rm /tmp/tmp*.pt
fi


if false; then
$my_python translate.py -gpu 0 -model onmt/tests/test_model.pt \
  -src data/src-val.txt -output onmt/tests/output_hyp.txt -beam 5 -batch_size 16

fi

############### TEST LANGUAGE MODEL
if false; then
rm data/data_lm/*.python

$my_python build_vocab.py \
    -config data/lm_data.yaml -save_data data/data_lm -share_vocab \
    -src_vocab data/data_lm/data.vocab.src -tgt_vocab data/data_lm/data.vocab.tgt \
    -overwrite true

$my_python train.py -config data/lm_data.yaml -save_model /tmp/tmp \
 -accum_count 2 -dec_layers 2 -rnn_size 64 -word_vec_size 64 -batch_size 256 \
 -encoder_type transformer_lm -decoder_type transformer_lm -share_embedding \
 -train_steps 2000 -max_generator_batches 4 -dropout 0.1 -normalization tokens \
 -share_vocab -transformer_ff 256 -max_grad_norm 0 -optim adam -decay_method noam \
 -learning_rate 2 -label_smoothing 0.1 -model_task lm -world_size 1 -gpu_ranks 0 \
 -attention_dropout 0.1 -heads 2 -position_encoding -param_init 0 -warmup_steps 100 \
 -param_init_glorot -adam_beta2 0.998 -src_vocab data/data_lm/data.vocab.src
#
mv /tmp/tmp*2000.pt onmt/tests/test_model_lm.pt
rm /tmp/tmp*.pt
fi
#
if false; then
$my_python translate.py -gpu 0 -model onmt/tests/test_model_lm.pt \
  -src data/src-val.txt -output onmt/tests/output_hyp.txt -beam 5 -batch_size 16

fi

