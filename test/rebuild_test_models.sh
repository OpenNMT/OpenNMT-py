# # Retrain the models used for CI.
# # Should be done rarely, indicates a major breaking change. 


python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/data -src_vocab_size 1000 -tgt_vocab_size 1000

python train.py -data data/data -save_model /tmp/tmp -gpuid 0 -rnn_size 100 -word_vec_size 50 -layers 1 -epochs 10 -optim adam  -learning_rate 0.001

mv /tmp/tmp*e10.pt test/test_model.pt

rm /tmp/tmp*.pt

python preprocess.py -train_src data/morph/src.train -train_tgt data/morph/tgt.train -valid_src data/morph/src.valid -valid_tgt data/morph/tgt.valid -save_data data/morph/data 

python train.py -data data/morph/data -save_model /tmp/tmp -gpuid 0 -rnn_size 400 -word_vec_size 100 -layers 1 -epochs 8

mv /tmp/tmp*e8.pt test/test_model2.pt

rm /tmp/tmp*.pt
