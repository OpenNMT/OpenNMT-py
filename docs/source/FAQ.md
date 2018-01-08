# FAQ

## How do I use Pretrained embeddings (e.g. GloVe)?

Using vocabularies from OpenNMT-py preprocessing outputs, `embeddings_to_torch.py` to generate encoder and decoder embeddings initialized with GloVe’s values.

the script is a slightly modified version of ylhsieh’s one2.

Usage:

```
embeddings_to_torch.py [-h] -emb_file EMB_FILE -output_file OUTPUT_FILE -dict_file DICT_FILE [-verbose]

emb_file: GloVe like embedding file i.e. CSV [word] [dim1] ... [dim_d]

output_file: a filename to save the output as PyTorch serialized tensors2

dict_file: dict output from OpenNMT-py preprocessing
```

Example


1) get GloVe files:

```
mkdir "glove_dir"
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d "glove_dir"
```

2) prepare data:

```
python preprocess.py \
-train_src data/train.src.txt \
-train_tgt data/train.tgt.txt \
-valid_src data/valid.src.txt \
-valid_tgt data/valid.tgt.txt \
-save_data data/data
```

3) prepare embeddings:

```
./tools/embeddings_to_torch.py -emb_file "glove_dir/glove.6B.100d.txt" \
-dict_file "data/data.vocab.pt" \
-output_file "data/embeddings"
```

4) train using pre-trained embeddings:

```
python train.py -save_model data/model \
-batch_size 64 \
-layers 2 \
-rnn_size 200 \
-word_vec_size 100 \
-pre_word_vecs_enc "data/embeddings.enc.pt" \
-pre_word_vecs_dec "data/embeddings.dec.pt" \
        -data data/data
```


## How do I use the Transformer model?

The transformer model is very sensitive to hyperparameters. To run it
effectively you need to set a bunch of different options that mimic the Google
setup.

```
python train.py -gpuid 0 -save_model models/ja-en-1M/ja-en -data database/ja-en -layers 6 \
        -rnn_size 512 -word_vec_size 512 -epochs 30 -max_grad_norm 0 -optim adam \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -dropout 0.1 -param_init 0 -warmup_steps 12000 -learning_rate 0.2 \
        -decay_method noam -label_smoothing 0.1 -adam_beta2 0.98 -batch_size 80 \
        -start_decay_at 31
```



## Do you support multi-gpu?

Currently our system does not support


