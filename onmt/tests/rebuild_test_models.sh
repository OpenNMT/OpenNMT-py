# # Retrain the models used for CI.
# # Should be done rarely, indicates a major breaking change. 
my_python=python3.5
export CUDA_VISIBLE_DEVICES=0,1,2,3
############### TEST regular RNN choose either -rnn_type LSTM / GRU / SRU and set input_feed 0 for SRU
if false; then
rm data/*.pt
$my_python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt \
  -valid_tgt data/tgt-val.txt -save_data data/data -src_vocab_size 1000 -tgt_vocab_size 1000 

$my_python train.py -data data/data -save_model onmt/tests/test_model -gpuid 0 -rnn_size 256 -word_vec_size 256 -layers 1 \
  -train_steps 10000 -valid_steps 1000 -save_checkpoint_steps 2000 -optim adam  -learning_rate 0.001 -rnn_type LSTM -input_feed 0
#-truncated_decoder 5 
#-label_smoothing 0.1
fi
#
# 2.28M Param - 
# 
############### TEST CNN 
if false; then
rm data/*.pt
$my_python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/data -src_vocab_size 1000 -tgt_vocab_size 1000 

$my_python train.py -data data/data -save_model onmt/tests/test_model -gpuid 0 -rnn_size 256 -word_vec_size 256 -layers 2 \
  -train_steps 10000 -valid_steps 1000 -save_checkpoint_steps 2000 -optim adam  -learning_rate 0.001 \
  -encoder_type cnn -decoder_type cnn
fi
#
# size256 - 1.76M Param - 
# 2x256 - 2.61M Param   - 
################# MORPH DATA
if false; then
rm data/morph/*.pt
$my_python preprocess.py -train_src data/morph/src.train -train_tgt data/morph/tgt.train -valid_src data/morph/src.valid \
  -valid_tgt data/morph/tgt.valid -save_data data/morph/data 

$my_python train.py -data data/morph/data -save_model onmt/tests/test_model2 -gpuid 0 -rnn_size 400 -word_vec_size 100 -layers 1 \
  -train_steps 10000 -valid_steps 1000 -save_checkpoint_steps 2000 -optim adam  -learning_rate 0.001
fi
############### TEST TRANSFORMER
if true; then
rm data/*.pt
$my_python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt \
  -valid_tgt data/tgt-val.txt -save_data data/data -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab

$my_python train.py -data data/data -save_model onmt/tests/test_model -batch_type tokens -batch_size 1024 -accum_count 1 \
 -layers 1 -rnn_size 8 -word_vec_size 8 -encoder_type transformer -decoder_type transformer -share_embedding \
 -gpuid 0 1 2 3 -max_generator_batches 4 -dropout 0.1 -normalization tokens \
 -max_grad_norm 0 -optim adam -decay_method noam -learning_rate 2 -label_smoothing 0.1 \
 -position_encoding -param_init 0 -warmup_steps 100 -param_init_glorot -adam_beta2 0.998 -seed 1111 \
 -train_steps 10000 -valid_steps 1000 -save_checkpoint_steps 2000
fi
#
# 3.41M Param - 
if false; then
$my_python translate.py -gpu 0 -model onmt/tests/test_model.pt \
  -src data/src-val.txt -output onmt/tests/output_hyp.txt -beam 5 -batch_size 16

fi


