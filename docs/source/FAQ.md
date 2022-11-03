
All the example YAML configurations are partial. To get an overview of what this YAML configuration is you can start by reading the [Quickstart](quickstart) section.

## How do I use Pretrained embeddings (e.g. GloVe)?

This is handled in the initial steps of the `onmt_train` execution.

Pretrained embeddings can be configured in the main YAML configuration file.

### Example

1. Get GloVe files:

```bash
mkdir "glove_dir"
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d "glove_dir"
```

2. Adapt the configuration:

```yaml
# <your_config>.yaml

<Your data config...>

...

# this means embeddings will be used for both encoder and decoder sides
both_embeddings: glove_dir/glove.6B.100d.txt
# to set src and tgt embeddings separately:
# src_embeddings: ...
# tgt_embeddings: ...

# supported types: GloVe, word2vec
embeddings_type: "GloVe"

# word_vec_size need to match with the pretrained embeddings dimensions
word_vec_size: 100

```

3. Train:

```bash
onmt_train -config <your_config>.yaml
```

Notes:

- the matched embeddings will be saved at `<save_data>.enc_embeddings.pt` and `<save_data>.dec_embeddings.pt`;
- additional flags `freeze_word_vecs_enc` and `freeze_word_vecs_dec` are available to freeze the embeddings.

## How do I use the Transformer model?

The transformer model is very sensitive to hyperparameters. To run it
effectively you need to set a bunch of different options that mimic the [Google](https://arxiv.org/abs/1706.03762) setup. We have confirmed the following configuration can replicate their WMT results.

```yaml
<data configuration>
...

# General opts
save_model: foo
save_checkpoint_steps: 10000
valid_steps: 10000
train_steps: 200000

# Batching
bucket_size: 32768
world_size: 4
gpu_ranks: [0, 1, 2, 3]
num_workers: 4
batch_type: "tokens"
batch_size: 4096
valid_batch_size: 8
max_generator_batches: 2
accum_count: [4]
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
position_encoding: true
enc_layers: 6
dec_layers: 6
heads: 8
hidden_size: 512
word_vec_size: 512
transformer_ff: 2048
dropout_steps: [0]
dropout: [0.1]
attention_dropout: [0.1]
```

Here are what the most important parameters mean:

* `param_init_glorot` & `param_init 0`: correct initialization of parameters;
* `position_encoding`: add sinusoidal position encoding to each embedding;
* `optim adam`, `decay_method noam`, `warmup_steps 8000`: use special learning rate;
* `batch_type tokens`, `normalization tokens`: batch and normalize based on number of tokens and not sentences;
* `accum_count 4`: compute gradients based on four batches;
* `label_smoothing 0.1`: use label smoothing loss.

## Do you support multi-gpu?

First you need to make sure you `export CUDA_VISIBLE_DEVICES=0,1,2,3`.

If you want to use GPU id 1 and 3 of your OS, you will need to `export CUDA_VISIBLE_DEVICES=1,3`

Both `-world_size` and `-gpu_ranks` need to be set. E.g. `-world_size 4 -gpu_ranks 0 1 2 3` will use 4 GPU on this node only.

**Warning - Deprecated**

Multi-node distributed training has not been properly re-implemented since OpenNMT-py 2.0.

If you want to use 2 nodes with 2 GPU each, you need to set `-master_ip` and `-master_port`, and

* `-world_size 4 -gpu_ranks 0 1`: on the first node
* `-world_size 4 -gpu_ranks 2 3`: on the second node
* `-accum_count 2`: This will accumulate over 2 batches before updating parameters.

If you use a regular network card (1 Gbps) then we suggest to use a higher `-accum_count` to minimize the inter-node communication.

**Note:**

In the legacy version, when training on several GPUs, you couldn't have them in 'Exclusive' compute mode (`nvidia-smi -c 3`).

The multi-gpu setup relied on a Producer/Consumer setup. This setup means there will be `2<n_gpu> + 1` processes spawned, with 2 processes per GPU, one for model training and one (Consumer) that hosts a `Queue` of batches that will be processed next. The additional process is the Producer, creating batches and sending them to the Consumers. This setup is beneficial for both wall time and memory, since it loads data shards 'in advance', and does not require to load it for each GPU process.

The new codebase allows GPUs to be in exclusive mode, because batches are moved to the device later in the process. Hence, there is no 'producer' process on each GPU.

## How can I ensemble Models at inference?

You can specify several models in the `onmt_translate` command line: `-model model1_seed1 model2_seed2`
Bear in mind that your models must share the same target vocabulary.

## How can I weight different corpora at training?

This is naturally embedded in the data configuration format introduced in OpenNMT-py 2.0. Each entry of the `data` configuration will have its own *weight*. When building batches, we'll sequentially take *weight* example from each corpus.

**Note**: don't worry about batch homogeneity/heterogeneity, the bucketing mechanism is here for that reason. Instead of building batches one at a time, we will load `bucket_size` examples, sort them by length, build batches and then yield them in a random order.

### Example

In the following example, we will sequentially sample 7 examples from *corpus_1*, and 3 examples from *corpus_2*, and so on:

```yaml
# <your_config>.yaml

...

# Corpus opts:
data:
    corpus_1:
        path_src: toy-ende/src-train1.txt
        path_tgt: toy-ende/tgt-train1.txt
        weight: 7
    corpus_2:
        path_src: toy-ende/src-train1.txt
        path_tgt: toy-ende/tgt-train1.txt
        weight: 3
    valid:
        path_src: toy-ende/src-val.txt
        path_tgt: toy-ende/tgt-val.txt
...

```

## How can I apply on-the-fly tokenization and subword regularization when training?

This is naturally embedded in the data configuration format introduced in OpenNMT-py 2.0. Each entry of the `data` configuration will have its own `transforms`. `transforms` basically is a `list` of functions that will be applied sequentially to the examples when read from file.

### Example

This example applies sentencepiece tokenization with `pyonmttok`, with `nbest=20` and `alpha=0.1`.

```yaml
# <your_config>.yaml

...

# Tokenization options
src_subword_type: sentencepiece
src_subword_model: examples/subword.spm.model
tgt_subword_type: sentencepiece
tgt_subword_model: examples/subword.spm.model

# Number of candidates for SentencePiece sampling
subword_nbest: 20
# Smoothing parameter for SentencePiece sampling
subword_alpha: 0.1
# Specific arguments for pyonmttok
src_onmttok_kwargs: "{'mode': 'none', 'spacer_annotate': True}"
tgt_onmttok_kwargs: "{'mode': 'none', 'spacer_annotate': True}"

# Corpus opts:
data:
    corpus_1:
        path_src: toy-ende/src-train1.txt
        path_tgt: toy-ende/tgt-train1.txt
        transforms: [onmt_tokenize]
        weight: 1
    valid:
        path_src: toy-ende/src-val.txt
        path_tgt: toy-ende/tgt-val.txt
        transforms: [onmt_tokenize]
...

```

Other tokenization methods and transforms are readily available. See the dedicated docs for more details.

## What are the readily available on-the-fly data transforms?

It's your lucky day! We already embedded several transforms that can be used easily.

Note: all the details about every flag and options for each transform can be found in the [train](#train) section.

### General purpose

#### Filter examples by length

Transform name: `filtertoolong`

Class: `onmt.transforms.misc.FilterTooLongTransform`

The following options can be added to the configuration :
- `src_seq_length`: maximum source sequence length;
- `tgt_seq_length`: maximum target sequence length.

#### Add custom prefix to examples

Transform name: `prefix`

Class: `onmt.transforms.misc.PrefixTransform`

For each dataset that the `prefix` transform is applied to, you can set the additional `src_prefix` and `tgt_prefix` parameters in its data configuration:

```yaml
data:
    corpus_1:
        path_src: toy-ende/src-train1.txt
        path_tgt: toy-ende/tgt-train1.txt
        transforms: [prefix]
        weight: 1
        src_prefix: __some_src_prefix__
        tgt_prefix: __some_tgt_prefix__
```

### Tokenization

Common options for the tokenization transforms are the following:

- `src_subword_model`: path of source side (or both if shared) subword model;
- `tgt_subword_model`: path of target side subword model;
- `src_subword_nbest`: number of candidates for subword regularization (sentencepiece), source side;
- `tgt_subword_nbest`: number of candidates for subword regularization (sentencepiece), target_side;
- `src_subword_alpha`: smoothing parameter for sentencepiece regularization / dropout probability for BPE, source side;
- `tgt_subword_alpha`: smoothing parameter for sentencepiece regularization / dropout probability for BPE, target side.

#### [OpenNMT Tokenizer](https://github.com/opennmt/Tokenizer)

Transform name: `onmt_tokenize`

Class: `onmt.transforms.tokenize.ONMTTokenizerTransform`

Additional options are available:
- `src_subword_type`: type of subword model for source side (from `["none", "sentencepiece", "bpe"]`);
- `tgt_subword_type`: type of subword model for target side (from `["none", "sentencepiece", "bpe"]`);
- `src_onmttok_kwargs`: additional kwargs for pyonmttok Tokenizer class, source side;
- `tgt_onmttok_kwargs`: additional kwargs for pyonmttok Tokenizer class, target side.

#### [SentencePiece](https://github.com/google/sentencepiece)

Transform name: `sentencepiece`

Class: `onmt.transforms.tokenize.SentencePieceTransform`

The `src_subword_model` and `tgt_subword_model` should be valid sentencepiece models.

#### BPE ([subword-nmt](https://github.com/rsennrich/subword-nmt))

Transform name: `bpe`

Class: `onmt.transforms.tokenize.BPETransform`

The `src_subword_model` and `tgt_subword_model` should be valid BPE models.

### BART-style noise

BART-style noise is composed of several parts, as described in [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461).

These different types of noise can be controlled with the following options:

- `permute_sent_ratio`: proportion of sentences to permute (default boundaries are ".", "?" and "!");
- `rotate_ratio`: proportion of inputs to permute;
- `insert_ratio`: proportion of additional random tokens to insert;
- `random_ratio`: proportion of tokens to replace with random;
- `mask_ratio`: proportion of words/subwords to mask;
- `mask_length`: length of masking window (from `["subword", "word", "span-poisson"]`);
- `poisson_lambda`: $\lambda$ value for Poisson distribution to sample span length (in the case of `mask_length` set to `span-poisson`);
- `replace_length`: when masking N tokens, replace with 0, 1, " "or N tokens. (set to -1 for N).

### SwitchOut and sampling

#### [SwitchOut](https://arxiv.org/abs/1808.07512)

Transform name: `switchout`

Class: `onmt.transforms.sampling.SwitchOutTransform`

Options:

- `switchout_temperature`: sampling temperature for SwitchOut.

#### Drop some tokens

Transform name: `tokendrop`

Class: `onmt.transforms.sampling.TokenDropTransform`

Options:

- `tokendrop_temperature`: sampling temperature for token deletion.

#### Mask some tokens

Transform name: `tokenmask`

Class: `onmt.transforms.sampling.TokenMaskTransform`

Options:

- `tokenmask_temperature`: sampling temperature for token masking.

## How can I create custom on-the-fly data transforms?

The code is easily extendable with custom transforms inheriting from the `Transform` base class.

You can for instance have a look at the `FilterTooLongTransform` class as a template:

```python
@register_transform(name='filtertoolong')
class FilterTooLongTransform(Transform):
    """Filter out sentence that are too long."""

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to this Transform."""
        group = parser.add_argument_group("Transform/Filter")
        group.add("--src_seq_length", "-src_seq_length", type=int, default=200,
                  help="Maximum source sequence length.")
        group.add("--tgt_seq_length", "-tgt_seq_length", type=int, default=200,
                  help="Maximum target sequence length.")

    def _parse_opts(self):
        self.src_seq_length = self.opts.src_seq_length
        self.tgt_seq_length = self.opts.tgt_seq_length

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Return None if too long else return as is."""
        if (len(example['src']) > self.src_seq_length or
                len(example['tgt']) > self.tgt_seq_length):
            if stats is not None:
                stats.update(FilterTooLongStats())
            return None
        else:
            return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}, {}={}'.format(
            'src_seq_length', self.src_seq_length,
            'tgt_seq_length', self.tgt_seq_length
        )
```

Methods:
- `add_options` allows to add custom options that would be necessary for the transform configuration;
- `_parse_opts` allows to parse options introduced in `add_options` when initialize;
- `apply` is where the transform happens;
- `_repr_args` is for clean logging purposes.

As you can see, there is the `@register_transform` wrapper before the class definition. This will allow for the class to be automatically detected (if put in the proper `transforms` folder) and usable in your training configurations through its `name` argument.

You could also collect statistics for your custom transform by creating a class inheriting `ObservableStats`:

```python
class FilterTooLongStats(ObservableStats):
    """Runing statistics for FilterTooLongTransform."""
    __slots__ = ["filtered"]

    def __init__(self):
        self.filtered = 1

    def update(self, other: "FilterTooLongStats"):
        self.filtered += other.filtered
```

NOTE:
- Add elements to keep track in the `__init__` and also `__slot__` to make it lightweight;
- Supply update logic in `update` method;
- (Optional) override `__str__` to change default log message format;
- Instantiate and passing the statistic object in the `apply` method of the corresponding transform class;
- statistics will be gathered per corpus per worker, but only first worker will report for its shard by default.

The `example` argument of `apply` is a `dict` of the form:
```
{
	"src": <source string>,
	"tgt": <target string>,
	"align": <alignment pharaoh string> # optional
}
```

This is defined in `onmt.inputters.corpus.ParallelCorpus.load`. This class is not easily extendable for now but it can be considered for future developments. For instance, we could create some `CustomParallelCorpus` class that would handle other kind of inputs.


## Can I get word alignments while translating?

### Raw alignments from averaging Transformer attention heads

Currently, we support producing word alignment while translating for Transformer based models. Using `-report_align` when calling `translate.py` will output the inferred alignments in Pharaoh format. Those alignments are computed from an argmax on the average of the attention heads of the *second to last* decoder layer. The resulting alignment src-tgt (Pharaoh) will be pasted to the translation sentence, separated by ` ||| `.
Note: The *second to last* default behaviour was empirically determined. It is not the same as the paper (they take the *penultimate* layer), probably because of slight differences in the architecture.

* alignments use the standard "Pharaoh format", where a pair `i-j` indicates the i<sub>th</sub> word of source language is aligned to j<sub>th</sub> word of target language.
* Example: {'src': 'das stimmt nicht !'; 'output': 'that is not true ! ||| 0-0 0-1 1-2 2-3 1-4 1-5 3-6'}
* Using the`-tgt` option when calling `translate.py`, we output alignments between the source and the gold target rather than the inferred target, assuming we're doing evaluation.
* To convert subword alignments to word alignments, or symetrize bidirectional alignments, please refer to the [lilt scripts](https://github.com/lilt/alignment-scripts).

### Supervised learning on a specific head

The quality of output alignments can be further improved by providing reference alignments while training. This will invoke multi-task learning on translation and alignment. This is an implementation based on the paper [Jointly Learning to Align and Translate with Transformer Models](https://arxiv.org/abs/1909.02074).

The data need to be preprocessed with the reference alignments in order to learn the supervised task.
The reference alignment file(s) can for instance be generated by [GIZA++](https://github.com/moses-smt/mgiza/) or [fast_align](https://github.com/clab/fast_align).

In order to learn the supervised task, you can set for each dataset the path of its alignment file in the YAML configuration file:

```yaml
<your_config>.yaml

...

# Corpus opts:
data:
    corpus_1:
        path_src: toy-ende/src-train1.txt
        path_tgt: toy-ende/tgt-train1.txt
        # src - tgt alignments in pharaoh format
        path_align: toy-ende/src-tgt.align
        transforms: []
        weight: 1
    valid:
        path_src: toy-ende/src-val.txt
        path_tgt: toy-ende/tgt-val.txt
        transforms: []

...
```

**Notes**:
- Most of the transforms are for now incompatible with the joint alignment learning pipeline, because most of them make modifications at the token level, hence alignments would be made invalid.
- There should be no blank lines in the alignment files provided.

Training options to learn such alignments are:

* `-lambda_align`: set the value > 0.0 to enable joint align training, the paper suggests 0.05;
* `-alignment_layer`: indicate the index of the decoder layer;
* `-alignment_heads`:  number of alignment heads for the alignment task - should be set to 1 for the supervised task, and preferably kept to default (or same as `num_heads`) for the average task;
* `-full_context_alignment`: do full context decoder pass (no future mask) when computing alignments. This will slow down the training (~12% in terms of tok/s) but will be beneficial to generate better alignment.


## How can I update a checkpoint's vocabulary?

New vocabulary can be used to continue training from a checkpoint. Existing vocabulary embeddings will be mapped to the new vocabulary, and new vocabulary tokens will be initialized as usual.

Run `onmt_build_vocab` as usual with the new dataset. New vocabulary files will be created.

Training options to perform vocabulary update are:

* `-update_vocab`: set this option
* `-reset_optim`: set the value to "states"
* `-train_from`: checkpoint path


## How can I use source word features?

Extra information can be added to the words in the source sentences by defining word features. 

Features should be defined in a separate file using blank spaces as a separator and with each row corresponding to a source sentence. An example of the input files:

data.src
```
however, according to the logs, she is hard-working.
```

feat.txt
```
A C C C C A A B
```

Prior tokenization is not necessary, features will be inferred by using the `FeatInferTransform` transform if tokenization has been applied.

No previous tokenization:
```
SRC: this is a test.
FEATS: A A A B
TOKENIZED SRC: this is a test ￭.
RESULT: A A A B <null>
```

Previously tokenized:
```
SRC: this is a test ￭.
FEATS: A A A B A
RESULT: A A A B A
```

**Notes**
- `FilterFeatsTransform` and `FeatInferTransform` are required in order to ensure the functionality.
- Not possible to do shared embeddings (at least with `feat_merge: concat` method)

Sample config file:

```
data:
    dummy:
        path_src: data/train/data.src
        path_tgt: data/train/data.tgt
        src_feats:
            feat_0: data/train/data.src.feat_0
            feat_1: data/train/data.src.feat_1
        transforms: [filterfeats, onmt_tokenize, inferfeats, filtertoolong]
        weight: 1
    valid:
        path_src: data/valid/data.src
        path_tgt: data/valid/data.tgt
        src_feats:
            feat_0: data/valid/data.src.feat_0
            feat_1: data/valid/data.src.feat_1
        transforms: [filterfeats, onmt_tokenize, inferfeats]

# Transform options
reversible_tokenization: "joiner"
prior_tokenization: true

# Vocab opts
src_vocab: exp/data.vocab.src
tgt_vocab: exp/data.vocab.tgt
src_feats_vocab: 
    feat_0: exp/data.vocab.feat_0
    feat_1: exp/data.vocab.feat_1
feat_merge: "sum"
```

During inference you can pass features by using the `--src_feats` argument. `src_feats` is expected to be a Python like dict, mapping feature names with their data file.

```
{'feat_0': '../data.txt.feats0', 'feat_1': '../data.txt.feats1'}
```

**Important note!** During inference, input sentence is expected to be tokenized. Therefore feature inferring should be handled prior to running the translate command. Example:

```bash
python translate.py -model model_step_10.pt -src ../data.txt.tok -output ../data.out --src_feats "{'feat_0': '../data.txt.feats0', 'feat_1': '../data.txt.feats1'}"
```

When using the Transformer architecture make sure the following options are appropriately set:

- `src_word_vec_size` and `tgt_word_vec_size` or `word_vec_size`
- `feat_merge`: how to handle features vecs
- `feat_vec_size` and maybe `feat_vec_exponent`


## How can I set up a translation server ?
A REST server was implemented to serve OpenNMT-py models. A discussion is opened on the OpenNMT forum: [discussion link](https://forum.opennmt.net/t/simple-opennmt-py-rest-server/1392).

### I. How it works?
---
The idea behind the translation server is to make a entry point for translation with multiple models. The server will receive natural text input, tokenize it, translate it following the decoding parameters, detokenize the result and return natural text output.

A server configuration file (`./available_models/conf.json`) is required. It contains the path of the model checkpoint, the path of tokenizer's data along with other inference parameters.

##### Configuration:
- `models_root`: (opt) folder containing model checkpoints, [default: `./available_models`]
- `models`: list of objects such as :
  - `id`: (opt) manually assign an id (int), [default: value from counter]
  - `name`: (opt) assing a name (str)
  - `model`: (required) path to checkpoint file i.e. `*.pt`
  - `timeout`: (opt) interval (seconds) before unloading, reset at each translation using the model
  - `load`: (opt) whether to load the model at start [default: False]
  - `on_timeout`: (opt) what to do on timeout: `unload` removes everything; `to_cpu` transfer the model to RAM (from GPU memory) this is faster to reload but takes RAM.
  - `opt`: (opt) dict of translation options (see method `translate_opts` in `./opts.py`)
  - `tokenizer`: (opt) set tokenizer options (if any), such as:
    - `type`: (str) value in `{sentencepiece, pyonmttok}`.
    - `model`: (str) path to tokenizer model
  - `ct2_translator_args` and `ct2_translate_batch_args`: (opt) [CTranslate2](https://github.com/OpenNMT/CTranslate2) parameters to use CTranslate2 inference engine. Parameters appearing simultaneously in `opt` and `ct2_(...)_args` must be identical.
  - `ct2_model`: (opt) CTranslate2 model path.


##### Example
```json
{
    "models_root": "./available_models",
    "models": [
        {   
            "id": 100,
            "model": "model_0.pt",
            "timeout": 600,
            "on_timeout": "to_cpu",
            "load": true,
            "opt": {
                "gpu": 0,
                "beam_size": 5
            },  
            "tokenizer": {
                "type": "sentencepiece",
                "model": "wmtenfr.model"
            }   
        },{ 
            "model": "model_0.light.pt",
            "timeout": -1, 
            "on_timeout": "unload",
            "model_root": "../other_models",
            "opt": {
                "batch_size": 1,
                "beam_size": 10
            }   
        }   
    ]   
}
```

### II. How to start the server without Docker ?
---
##### 0. Get the code
The translation server has been merged into onmt-py `master` branch.   
Keep in line with master for last fix / improvements.
##### 1. Install `flask`
```bash
pip install flask
```

##### 2. Put some models
```bash
mkdir available_models/
cp $path_to_my_model available_models
```

##### 3. Start the server
```bash
export IP="0.0.0.0"
export PORT=5000
export URL_ROOT="/translator"
export CONFIG="./available_models/conf.json"

# NOTE that these parameters are optionnal
# here, we explicitely set to default values
python server.py --ip $IP --port $PORT --url_root $URL_ROOT --config $CONFIG
```

### III. How to start the server with Docker ?
---

1. Add the following libraries a requirement file `requirements.docker.txt`.
```
ConfigArgParse==1.2.3
Flask==1.1.2
Flask-Cors==3.0.10
pyonmttok==1.22.1
torchtext==0.4.0
waitress==1.4.4
```

2. Create a `Dockerfile`
```docker
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
WORKDIR /usr/src/app

COPY requirements.docker.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# You can copy or use a docker volume, especially for the model and config data
COPY server.py ./
COPY tools ./tools
COPY available_models ./available_models
COPY onmt ./onmt

CMD ["python", "./server.py"]
```

3. Build the image and run container
```bash
docker build -t opennmt_server .
docker run -it --rm -p 5000:5000 opennmt_server
```

### IV. How to use the API ?
----
This section contains a fex examples of the API. For details on all routes, see `./bin/server.py`.
##### 0. Set the hostname
```bash
export HOST="127.0.0.1"
```
##### 1. List models

```bash
curl http://$HOST:$PORT$URL_ROOT/models
```

**Result (example):**
```json
{
  "available": [
    "wmt14.en-de_acc_69.22_ppl_4.33_e9.pt",
    "wmt14.en-de_acc_69.22_ppl_4.33_e9.light.pt"
  ],
  "loaded": []
}
```
##### 2. Translate
(this example involves subwords)
```bash
curl -i -X POST -H "Content-Type: application/json" \
    -d '[{"src": "this is a test for model 0", "id": 0}]' \
    http://$HOST:$PORT$URL_ROOT/translate

```
**Result:**
```json
{
  "model_id": 0,
  "result": "\u2581die \u2581Formen kant en \u2581( K \u00f6r ner ) \u2581des \u2581Stahl g u\u00df form .\n",
  "status": "ok",
  "time": {
    "total": 8.510261535644531,
    "translation": 8.509992599487305,
    "writing_src": 0.0002689361572265625
  }
}
```
