
# Translation

This example is for training for the [WMT'14 English to German news translation task](https://www.statmt.org/wmt14/translation-task.html). It will use on the fly tokenization with [sentencepiece](https://github.com/google/sentencepiece) and [sacrebleu](https://github.com/mjpost/sacrebleu) for evaluation.


## Step 0: Download the data and prepare the subwords model

Preliminary steps are defined in the [`examples/scripts/prepare_wmt_data.sh`](https://github.com/OpenNMT/OpenNMT-py/tree/master/examples/scripts/prepare_wmt_data.sh). The following command will download the necessary datasets, and prepare a sentencepiece model:
```bash
chmod u+x prepare_wmt_data.sh
./prepare_wmt_data.sh
```

Note: you should have installed [sentencepiece](https://github.com/google/sentencepiece) binaries before running this script.

## Step 1. Build the vocabulary.

We need to setup the desired configuration with 1. the data 2. the tokenization options:

```yaml
# wmt14_en_de.yaml
save_data: data/wmt/run/example
## Where the vocab(s) will be written
src_vocab: data/wmt/run/example.vocab.src
tgt_vocab: data/wmt/run/example.vocab.tgt

# Corpus opts:
data:
    commoncrawl:
        path_src: data/wmt/commoncrawl.de-en.en
        path_tgt: data/wmt/commoncrawl.de-en.de
        transforms: [sentencepiece, filtertoolong]
        weight: 23
    europarl:
        path_src: data/wmt/europarl-v7.de-en.en
        path_tgt: data/wmt/europarl-v7.de-en.de
        transforms: [sentencepiece, filtertoolong]
        weight: 19
    news_commentary:
        path_src: data/wmt/news-commentary-v11.de-en.en
        path_tgt: data/wmt/news-commentary-v11.de-en.de
        transforms: [sentencepiece, filtertoolong]
        weight: 3
    valid:
        path_src: data/wmt/valid.en
        path_tgt: data/wmt/valid.de
        transforms: [sentencepiece]

### Transform related opts:
#### Subword
src_subword_model: data/wmt/wmtende.model
tgt_subword_model: data/wmt/wmtende.model
src_subword_nbest: 1
src_subword_alpha: 0.0
tgt_subword_nbest: 1
tgt_subword_alpha: 0.0
#### Filter
src_seq_length: 150
tgt_seq_length: 150

# silently ignore empty lines in the data
skip_empty_level: silent

```

Then we can execute the vocabulary building script. Let's set `-n_sample` to `-1` to compute the vocabulary over the whole corpora:

```bash
onmt_build_vocab -config wmt14_en_de.yaml -n_sample -1
```

## Step 2: Train the model

We need to add the following parameters to the YAML configuration:

```yaml
...

# General opts
save_model: data/wmt/run/model
keep_checkpoint: 50
save_checkpoint_steps: 5000
average_decay: 0.0005
seed: 1234
report_every: 100
train_steps: 100000
valid_steps: 5000

# Batching
queue_size: 10000
bucket_size: 32768
world_size: 2
gpu_ranks: [0, 1]
batch_type: "tokens"
batch_size: 4096
valid_batch_size: 16
batch_size_multiple: 1
max_generator_batches: 0
accum_count: [3]
accum_steps: [0]

# Optimization
model_dtype: "fp32"
optim: "adam"
learning_rate: 2
warmup_steps: 8000
decay_method: "noam"
adam_beta2: 0.998
max_grad_norm: 0
label_smoothing: 0.1
param_init: 0
param_init_glorot: true
normalization: "tokens"

# Model
encoder_type: transformer
decoder_type: transformer
enc_layers: 6
dec_layers: 6
heads: 8
rnn_size: 512
word_vec_size: 512
transformer_ff: 2048
dropout_steps: [0]
dropout: [0.1]
attention_dropout: [0.1]
share_decoder_embeddings: true
share_embeddings: true
```

## Step 3: Translate and evaluate

We need to tokenize the testset with the same sentencepiece model as used in training:

```bash
spm_encode --model=data/wmt/wmtende.model \
    < data/wmt/test.en \
    > data/wmt/test.en.sp
spm_encode --model=data/wmt/wmtende.model \
    < data/wmt/test.de \
    > data/wmt/test.de.sp
```

We can translate the testset with the following command:

```bash
for checkpoint in data/wmt/run/model_step*.pt; do
    echo "# Translating with checkpoint $checkpoint"
    base=$(basename $checkpoint)
    onmt_translate \
        -gpu 0 \
        -batch_size 16384 -batch_type tokens \
        -beam_size 5 \
        -model $checkpoint \
        -src data/wmt/test.en.sp \
        -tgt data/wmt/test.de.sp \
        -output data/wmt/test.de.hyp_${base%.*}.sp
done
```

Prior to evaluation, we need to detokenize the hypothesis:

```bash
for checkpoint in data/wmt/run/model_step*.pt; do
    base=$(basename $checkpoint)
    spm_decode \
        -model=data/wmt/wmtende.model \
        -input_format=piece \
        < data/wmt/test.de.hyp_${base%.*}.sp \
        > data/wmt/test.de.hyp_${base%.*}
done
```


Finally, we can compute detokenized BLEU with `sacrebleu`:

```bash
for checkpoint in data/wmt/run/model_step*.pt; do
    echo "$checkpoint"
    base=$(basename $checkpoint)
    sacrebleu data/wmt/test.de < data/wmt/test.de.hyp_${base%.*}
done
```
