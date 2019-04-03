# FAQ

## How do I use Pretrained embeddings (e.g. GloVe)?

Using vocabularies from OpenNMT-py preprocessing outputs, `embeddings_to_torch.py` to generate encoder and decoder embeddings initialized with GloVes values.

the script is a slightly modified version of ylhsiehs one2.

Usage:

```
embeddings_to_torch.py [-h] [-emb_file_both EMB_FILE_BOTH]
                       [-emb_file_enc EMB_FILE_ENC]
                       [-emb_file_dec EMB_FILE_DEC] -output_file
                       OUTPUT_FILE -dict_file DICT_FILE [-verbose]
                       [-skip_lines SKIP_LINES]
                       [-type {GloVe,word2vec}]
```
Run embeddings_to_torch.py -h for more usagecomplete info.

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
./tools/embeddings_to_torch.py -emb_file_both "glove_dir/glove.6B.100d.txt" \
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


## How do I use the Transformer model? Do you support multi-gpu?

The transformer model is very sensitive to hyperparameters. To run it
effectively you need to set a bunch of different options that mimic the Google
setup. We have confirmed the following command can replicate their WMT results.

```
python  train.py -data /tmp/de2/data -save_model /tmp/extra \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 \
        -world_size 4 -gpu_ranks 0 1 2 3 
```

Here are what each of the parameters mean:

* `param_init_glorot` `-param_init 0`: correct initialization of parameters
* `position_encoding`: add sinusoidal position encoding to each embedding
* `optim adam`, `decay_method noam`, `warmup_steps 8000`: use special learning rate.
* `batch_type tokens`, `normalization tokens`, `accum_count 4`: batch and normalize based on number of tokens and not sentences. Compute gradients based on four batches. 
- `label_smoothing 0.1`: use label smoothing loss. 

Multi GPU settings
First you need to make sure you export CUDA_VISIBLE_DEVICES=0,1,2,3
If you want to use GPU id 1 and 3 of your OS, you will need to export CUDA_VISIBLE_DEVICES=1,3
* `world_size 4 gpu_ranks 0 1 2 3`: This will use 4 GPU on this node only.

If you want to use 2 nodes with 2 GPU each, you need to set -master_ip and master_port, and
* `world_size 4 gpu_ranks 0 1`: on the first node
* `world_size 4 gpu_ranks 2 3`: on the second node
* `accum_count 2`: This will accumulate over 2 batches before updating parameters.

if you use a regular network card (1 Gbps) then we suggest to use a higher accum_count to minimize the inter-node communication.

## How can I ensemble Models at inference?

You can specify several models in the translate.py command line: -model model1_seed1 model2_seed2
Bear in mind that your models must share the same traget vocabulary.


