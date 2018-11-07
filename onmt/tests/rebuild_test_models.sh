# # Retrain the models used for CI.
# # Should be done rarely, indicates a major breaking change. 
my_python=python

############### TEST regular RNN choose either -rnn_type LSTM / GRU / SRU and set input_feed 0 for SRU
if true; then
rm data/*.pt
$my_python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/data -src_vocab_size 1000 -tgt_vocab_size 1000 

$my_python train.py -data data/data -save_model tmp -world_size 1 -gpu_ranks 0 -rnn_size 256 -word_vec_size 256 -layers 1 -train_steps 10000 -optim adam  -learning_rate 0.001 -rnn_type LSTM -input_feed 0
#-truncated_decoder 5 
#-label_smoothing 0.1

mv tmp*e10.pt onmt/tests/test_model.pt
rm tmp*.pt
fi
#
# 
############### TEST CNN 
if false; then
rm data/*.pt
$my_python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/data -src_vocab_size 1000 -tgt_vocab_size 1000 

$my_python train.py -data data/data -save_model /tmp/tmp -world_size 1 -gpu_ranks 0 -rnn_size 256 -word_vec_size 256 -layers 2 -train_steps 10000 -optim adam  -learning_rate 0.001 -encoder_type cnn -decoder_type cnn


mv /tmp/tmp*e10.pt onmt/tests/test_model.pt

rm /tmp/tmp*.pt
fi
#
################# MORPH DATA
if true; then
rm data/morph/*.pt
$my_python preprocess.py -train_src data/morph/src.train -train_tgt data/morph/tgt.train -valid_src data/morph/src.valid -valid_tgt data/morph/tgt.valid -save_data data/morph/data 

$my_python train.py -data data/morph/data -save_model tmp -world_size 1 -gpu_ranks 0 -rnn_size 400 -word_vec_size 100 -layers 1 -train_steps 8000 -optim adam  -learning_rate 0.001


mv tmp*e8.pt onmt/tests/test_model2.pt

rm tmp*.pt
fi
############### TEST TRANSFORMER
if false; then
rm data/*.pt
$my_python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/data -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab


$my_python train.py -data data/data -save_model /tmp/tmp -batch_type tokens -batch_size 1024 -accum_count 4 \
 -layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer -share_embedding \
 -train_steps 10000 -world_size 1 -gpu_ranks 0 -max_generator_batches 4 -dropout 0.1 -normalization tokens \
 -max_grad_norm 0 -optim adam -decay_method noam -learning_rate 2 -label_smoothing 0.1 \
 -position_encoding -param_init 0 -warmup_steps 100 -param_init_glorot -adam_beta2 0.998
#
mv /tmp/tmp*e10.pt onmt/tests/test_model.pt
rm /tmp/tmp*.pt
fi
#
if false; then
$my_python translate.py -gpu 0 -model onmt/tests/test_model.pt \
  -src data/src-val.txt -output onmt/tests/output_hyp.txt -beam 5 -batch_size 16

fi


