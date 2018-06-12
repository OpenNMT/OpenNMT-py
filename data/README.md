



> python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/data -src_vocab_size 1000 -tgt_vocab_size 1000

> python train.py -data data/data -save_model /n/rush_lab/data/tmp_ -gpuid 0 -rnn_size 100 -word_vec_size 50 -layers 1 \
  -train_steps 10000 -optim adam  -learning_rate 0.001
