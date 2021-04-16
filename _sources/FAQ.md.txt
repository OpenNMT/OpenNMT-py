
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
queue_size: 10000
bucket_size: 32768
world_size: 4
gpu_ranks: [0, 1, 2, 3]
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
rnn_size: 512
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

Multi-node distributed training is not properly implemented in OpenNMT-py 2.0 yet.

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

**Note**: don't worry about batch homogeneity/heterogeneity, the pooling mechanism is here for that reason. Instead of building batches one at a time, we will load `pool_factor` of batches worth of examples, sort them by length, build batches and then yield them in a random order.

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

Class: `onmt.transforms.misc.ONMTTokenizerTransform`

Additional options are available:
- `src_subword_type`: type of subword model for source side (from `["none", "sentencepiece", "bpe"]`);
- `tgt_subword_type`: type of subword model for target side (from `["none", "sentencepiece", "bpe"]`);
- `src_onmttok_kwargs`: additional kwargs for pyonmttok Tokenizer class, source side;
- `tgt_onmttok_kwargs`: additional kwargs for pyonmttok Tokenizer class, target side.

#### [SentencePiece](https://github.com/google/sentencepiece)

Transform name: `sentencepiece`

Class: `onmt.transforms.misc.SentencePieceTransform`

The `src_subword_model` and `tgt_subword_model` should be valid sentencepiece models.

#### BPE ([subword-nmt](https://github.com/rsennrich/subword-nmt))

Transform name: `bpe`

Class: `onmt.transforms.misc.BPETransform`

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

Class: `onmt.transforms.misc.SwitchOutTransform`

Options:

- `switchout_temperature`: sampling temperature for SwitchOut.

#### Drop some tokens

Transform name: `tokendrop`

Class: `onmt.transforms.misc.TokenDropTransform`

Options:

- `tokendrop_temperature`: sampling temperature for token deletion.

#### Mask some tokens

Transform name: `tokenmask`

Class: `onmt.transforms.misc.TokenMaskTransform`

Options:

- `tokenmask_temperature`: sampling temperature for token masking.

## How can I create custom on-the-fly data transforms?

The code is easily extendable with custom transforms inheriting from the `Transform` base class.

You can for instance have a look at the `FilterTooLongTransform` class as a template:

```python
@register_transform(name='filtertoolong')
class FilterTooLongTransform(Transform):
    """Filter out sentence that are too long."""

    def __init__(self, opts):
        super().__init__(opts)
        self.src_seq_length = opts.src_seq_length
        self.tgt_seq_length = opts.tgt_seq_length

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to this Transform."""
        group = parser.add_argument_group("Transform/Filter")
        group.add("--src_seq_length", "-src_seq_length", type=int, default=200,
                  help="Maximum source sequence length.")
        group.add("--tgt_seq_length", "-tgt_seq_length", type=int, default=200,
                  help="Maximum target sequence length.")

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Return None if too long else return as is."""
        if (len(example['src']) > self.src_seq_length or
                len(example['tgt']) > self.tgt_seq_length):
            if stats is not None:
                stats.filter_too_long()
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
- `apply` is where the transform happens;
- `_repr_args` is for clean logging purposes.

As you can see, there is the `@register_transform` wrapper before the class definition. This will allow for the class to be automatically detected (if put in the proper `transforms` folder) and usable in your training configurations through its `name` argument.

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
