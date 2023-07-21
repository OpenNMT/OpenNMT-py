
All the example YAML configurations are partial. To get an overview of what this YAML configuration is you can start by reading the [Quickstart](quickstart) section.

Also you can have a look at this: [Tutorial](https://github.com/ymoslem/OpenNMT-Tutorial)

## How do I use my v2 models in v3 ?

As a reminder, OpenNMT-py v2.x used to rely on Torchtext 0.5

This torchtext version used "Fields", "RawFields", "MultiFields" which were deprecated in torchtext versions > 0.5. In order to convert old models we have to mimic those old Class and as a result you need to install a newer version of torchtext.

If you use pytorch 1.12.1 then install torchtext 0.13

If you use pytorch 1.13.0 then install torchtext 0.14

After the conversion you can eliminate completely torchtext.

Conversion is perfomed using the following script:

python tools/convertv2_v3.py -v2model myoldmodel.pt -v3model newmodel.pt

The new checkpoint will no longer have a "fields" key, replaced by "vocab"
Some model options are modified as follow:

* `rnn_size` is now `hidden_size`
* `enc_rnn_size` is now `enc_hid_size`
* `dec_rnn_size` is now `dec_hid_size`

A new key `add_qkvbias` is set to `true` for old models.

New models will be trained by default with `false`

Special note for GPT2 type Language Model trained with v2

The special tokens of LM in v2 where not in line with NMT type models.
Only 3 special tokens (`<unk>`, `<blank>`, `</s>`) were in the the vocab.
For v3, we aligned LM and NMT models.
You need to update the vocab by training from v2 converted to v3 checkpoint, with the update_vocab flag.
It will make the vocab consistent with v3 structure.


## How do I train the Transformer model?

The transformer model is very sensitive to hyperparameters. To run it
effectively you need to set different options that mimic the [Google](https://arxiv.org/abs/1706.03762) setup. We have confirmed the following configuration can replicate their WMT results.

Please have a look at the Example of WMT17 EN-DE.

```yaml
<data configuration>
...

# General opts
save_model: mybasemodel
save_checkpoint_steps: 10000
valid_steps: 10000
train_steps: 200000

# Batching
bucket_size: 262144
world_size: 4
gpu_ranks: [0, 1, 2, 3]
num_workers: 2
batch_type: "tokens"
batch_size: 4096
valid_batch_size: 2048
accum_count: [4]
accum_steps: [0]

# Optimization
model_dtype: "fp16"
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

## Performance tips

* use `fp16`
* use `batch_size_multiple` 8
* use `vocab_size_multiple` 8
* Depending on the number of GPU use num_workers 4 (for 1 GPU) or 2 (for multiple GPU)
* To avoid averaging checkpoints you can use the "during training" average decay system.
* If you train a transformer we support `max_relative_positions` (use 20) instead of position_encoding. 
* for very fast inference convert your model to CTranslate2 format.

## Position encoding: Absolute vs Relative vs Rotary Embeddings vs Alibi

The basic feature is absolute position encoding stemming from the original Transformer Paper.
However, even with this, we can use SinusoidalInterleaved (default OpenNMT-py) or SinusoidalConcat (default Fairseq imported models)
* `position_encoding: true`
* `position_encoding_type: 'SinusoidalInterleaved'`
Do not forget to set also `param_init_glorot: true`

If you prefer to use relative position encoding, we support 3 modes:
* "Shaw": https://arxiv.org/abs/1803.02155 - you need to set `max_relative_positions: N` where N > 1 (use 16, 20, 32) see paper.
* "Rope" Rotary Embeddings: https://arxiv.org/abs/2104.09864 - you need to set `max_relative_positions: -1`
* "Alibi" (used by MPT-7B for example) https://arxiv.org/abs/2108.12409 - you need to set `max_relative_positions: -2`

In both cases, it is necessary to set `position_encoding: false`

In a nutshell, at the time if this writing (v3.1) absolute position encoding is managed in the Embeddings module, whereas
the relative position encoding is managed directly in the multi-head self-attention module.

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

## What special tokens does OpenNMT-py use?

In the v2, special tokens were different for SEQ2SEQ and LM:
LM was BOS, PAD, EOS with IDs (0, 1, 2) and the first vocab token started at id=3
SEQ2SEQ was UNK, PAD, BOS, EOS with IDs (0, 1, 2, 3) and first vocab token started at id=4

In v3 we changed this behavior to align things:
    group.add(
        "--default_specials",
        "-default_specilas",
        nargs="+",
        type=str,
        default=[
            DefaultTokens.UNK,
            DefaultTokens.PAD,
            DefaultTokens.BOS,
            DefaultTokens.EOS,
        ])

When we train a SEQ2SEQ model we use:
SRC: srctok1 srctok2 srctok3 .... srctokn
TGT: BOS tgttok1 tgttok2 ..... tgttokm EOS
But when training a LM
SRC: BOS srctok1 srctok2 srctok3 .... srctokn
TGT: srctok1 srctok2 srctok3 .... srctokn EOS

Having said that, sometimes we need to finetune models (eg: NLLB-200, Llama, ...) with existing vocab
and special tokens are not the same.

ex with NLLB-200
BOS id=0
PAD id=1
EOS id=2
UNK id=3
And the decoder start token is EOS (</s>) which means in fact that the BOS is never used.
At training, TGT needs to start with EOS instead of BOS in the default OpenNMT-py config.

Example of Llama
UNK id=0
BOS id=1
EOS id=2
There was no PAD but to avoid conflicts we forced PAD id=3 (which was token '<0x00>' in the original llama tokenizer)



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

At inference if you want to use the target prefix feature to prefix your target segment with a unique prefix (as opposed to a target prefix coming from a line-by-line file)
you need to set your yaml file as follow (example given with a target language as in the NLLB-200 case):
``` yaml
tgt_prefix: "spa_Latn" 
tgt_file_prefix: true
```

#### Add custom suffix to examples

Transform name: `suffix`

Class: `onmt.transforms.misc.SuffixTransform`

For each dataset that the `suffix` transform is applied to, you can set the additional `src_suffix` and `tgt_suffix` parameters in its data configuration:

```yaml
data:
    corpus_1:
        path_src: toy-ende/src-train1.txt
        path_tgt: toy-ende/tgt-train1.txt
        transforms: [suffix]
        weight: 1
        src_suffix: __some_src_suffix__
        tgt_suffix: __some_tgt_suffix__
```

#### Convert examples to uppercase

Transform name: `uppercase`

Class: `onmt.transforms.uppercase.UpperCaseTransform`

Converts source and target (if present) examples to uppercase so the model can learn better to translate
sentences in all caps. This transform normalizes the examples so the uppercased strings are stripped from
any diacritics and accents. Usually this is desirable for most languages, although there are few exceptions.

The following option can be added to the main configuration (same ratio for all dataset with this transform):
- `upper_corpus_ratio`: ratio of the corpus that will be transformed to uppercase (default: 0.01);

#### Normalize punctuation

Transform name: `normalize`

Class: `onmt.transforms.normalize.NormalizeTransform`

Normalizes source and target (if present) examples using the same rules as Moses punctuation normalizer.

The following options can be added to the configuration of each dataset:
- `src_lang`: en, de, cz/cs, fr (default='')
- `tgt_lang`: en, de, cz/cs, fr (default='')
- `penn`: Penn substitution (default=True)
- `norm_quote_commas`: Normalize quotations and commas (default=True)
- `norm_numbers`: Normalize numbers (default=True)
- `pre_replace_unicode_punct`: Replace unicode punct (default=False)
- `post_remove_control_chars`: Remove control chars (default=False)

#### Clean dataset

Transform name: `clean`

Class: `onmt.transforms.clean.CleanTransform`

Cleans source and target (if present) examples using a set of rules.

The following options can be added to the configuration of each dataset:
- `src_eq_tgt`: Remove example when source=target (default=True)
- `same_char`: Remove example if the same char is repeated 4 times (default=True)
- `same_word`: Remove example if the same word is repeated 3 times (default=True)
- `script_ok`: Remove example which contains chars that do not belong to these scripts (default=['Latin', 'Common'])
- `script_nok`: Remove example which contains chars that belong to these scripts  (default=[])
- `src_tgt_ratio`: Remove example for which src/tgt ration is <1/ratio or >ratio (default=2)
- `avg_tok_min`: Remove example for which the average token length is < X (default=3)
- `avg_tok_max`: Remove example for which the average token length is > X (default=20)
- `lang_id`: Remove example for which detected language is not in [] (default=['en', 'fr'])

#### Context / Doc aware transform

Transform name: `docify`

Class: `onmt.transforms.docify.DocifyTransform`

Concatenates several segments into one, separated with a delimiter.

Pre-requisite:

Dataset must be "Docs" separated by an empty line which will make clear a story ends at this empty line.

The following options can be added to the main configuration (same options for all dataset with this transform):
- `doc_length`: max token to be concatenated (default=200)
- `max_context`: number of delimiter (default=1 , ie 2 segments concatenated)

When working with several workers, this require some precaution in order to make sure "doc" are read linearly.

`max_context + 1` needs to be a multiple of `stride` = `Number of gpu x num_workers`

Example: `max_context=1` and 1 GPU, then num_workers must be 2 or 4.


#### Augment source segments with fuzzy matches for Neural Fuzzy Repair

Transform name: `fuzzymatch`

Class: `onmt.transforms.fuzzymatch.FuzzyMatchTransform`

Augments source segments with fuzzy matches for Neural Fuzzy Repair, as described in [Neural Fuzzy Repair: Integrating Fuzzy Matches into Neural Machine Translation](https://aclanthology.org/P19-1175). Currently, the transform augments source segments with only a single fuzzy match.
The Translation Memory (TM) format should be a flat text file, with each line containing the source and the target segment separated by a delimiter. As fuzzy matching during training is computational intensive, we offer some advice to achieve good performance and minimize overhead:

- Depending on your system's specs, you may have to experiment with the options `bucket_size`, `bucket_size_init`, and `bucket_size_increment`;
- You should increase the `num_workers` and `prefetch_factor` so your GPU does not have to wait for the batches to be augmented with fuzzy matches;
- Try to use a sensible Translation Memory size. 200k-250k translation units should be enough for yielding a sufficient number of matches;
- Although the transform performs some basic filtering both in the TM and in the corpus for very short or very long segments, some examples may still be long enough, so you should increase a bit the `src_seq_length`;
- Currently, when using `n_sample`, examples are always processed one by one and not in batches.

The following options can be added to the main configuration (valid for all datasets using this transform):
- `tm_path`: The path to the Translation Memory text file;
- `fuzzy_corpus_ratio`: Ratio of corpus to augment with fuzzy matches (default: 0.1);
- `fuzzy_threshold`: The fuzzy matching threshold (default: 70);
- `tm_delimiter`: The delimiter used in the flat text TM (default: "\t");
- `fuzzy_token`: The fuzzy token to be added with the matches (default: "｟fuzzy｠");
- `fuzzymatch_min_length`: Min length for TM entries and examples to match (default: 4);
- `fuzzymatch_max_length`: Max length for TM entries and examples to match (default: 70).

#### Augment source and target segments with inline tags

Transform name: `inlinetags`

Class: `onmt.transforms.inlinetags.InlineTagsTransform`

Augments source and target segments with inline tags (placeholders). The transform adds 2 kind of tags, paired tags (an opening and a closing tag) and isolated (standalone) tags, and requires a tab-delimited dictionary text file with source and target terms and phrases. A dictionary with 20-30k entries is recommended. User-defined tags must include the number placeholder #, e.g. "｟user_start_tag_#｠".

The following options can be added to the main configuration (valid for all datasets using this transform):
- `tags_dictionary_path`: The path to the dictionary text file;
- `tags_corpus_ratio`: Ratio of corpus to augment with inline tags (default: 0.1);
- `max_tags`: Maximum number of tags that can be added to a single sentence. (default: 12);
- `paired_stag`: The format of an opening paired inline tag. Must include the character # (default: "｟ph_#_beg｠");
- `paired_etag`: The format of a closing paired inline tag. Must include the character # (default: "｟ph_#_end｠");
- `isolated_tag`: The format of an isolated inline tag. Must include the character # (default: "｟ph_#_std｠");
- `src_delimiter`: Any special token used for augmented src sentences (default: "｟fuzzy｠");

#### Make the model learn to use terminology

Transform name: `terminology`

Class: `onmt.transforms.terminology.TerminologyTransform`

Augments source segments with terms so the model can learn to use user-provided terms at inference. It requires a dictionary with source and target terms, delimited with a tab. The transform uses Spacy's lemmatization facilities in order to a) solve the word inflection problem when searching for terms in any form, and b) make the model inflect correctly most target terms at inference. The lemmatization is applied at the dictionary entries and also at the source and target examples, and the term searches during training are performed on the lemmatized examples.
 The format of a processed segment augmented with terms is as follows:
`This is an ｟src_term_start｠ augmented ｟tgt_term_start｠ target_lemma_for_augmented ｟tgt_term_end｠ example.`
The following options can be added to the main configuration (valid for all datasets using this transform):
- `termbase_path`: The path to the dictionary text file;
- `src_spacy_language_model`: Name of the spacy language model for the source corpus;
- `tgt_spacy_language_model`: Name of the spacy language model for the target corpus;
- `term_corpus_ratio`: Ratio of corpus to augment with terms # (default: 0.3);
- `term_example_ratio`: Max terms allowed in an example # (default: 0.2);
- `src_term_stoken`: The source term start token # (default: "｟src_term_start｠");
- `tgt_term_stoken`: The target term start token # (default: "｟tgt_term_start｠");
- `tgt_term_etoken`: The target term end token # (default: "｟tgt_term_end｠");
- `term_source_delimiter`: Any special token used for augmented src sentences. The default is the fuzzy token used in the FuzzyMatch transform # (default: "｟fuzzy｠");

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

#### [BPE subword-nmt](https://github.com/rsennrich/subword-nmt)

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


## How to use LoRa and 8bit loading to finetune a big model ?

Cf paper: [LoRa](https://arxiv.org/abs/2106.09685)

LoRa is a mechanism that helps to finetune bigger model on a single GPU card by limiting the anmount of VRAM needed.
The principle is to make only a few layers trainable (hence reducing the amount of required memory especially for the Adam optimizer)

You need to train_from a model (for instance NLLB-200 3.3B) and use the following options:

* `lora_layers: ['linear_values', 'linear_query']` these are the two layers of the Self-Attention module the paper recommend to make trainable.
* `lora_rank: 2`
* `lora_dropout: 0.1` or any value you can test
* `lora_alpha: 1` or any value you can test
* `lora_embedding: true` makes Embeddings LoRa compatible, hence trainable in the case you use `update_vocab: true` or if you want to finetune Embeddings as well.

Bitsandbytes enables quantization of Linear layers. For more information: https://github.com/TimDettmers/bitsandbytes
Also you can read the blog post here: https://huggingface.co/blog/hf-bitsandbytes-integration

You need to add the following option:

* `quant_layers: ['w_1', 'w_2', 'linear_values', 'linear_query']`
* `quant_type: ['bnb_NF4']`

You can for instane quantize the layers of the PositionWise Feed-Forward from the Encoder/Decoder and the key/query/values/final from the Multi-head attention.
Choices for quantization are ["bnb_8bit", "bnb_FP4", "bnb_NF4"]

## How to use gradient checkpointing when dealing with a big model ?

* `use_ckpting: ["ffn", "mha", "lora"]`

Be carefull, the module that you use checkpointing needs to have gradients.


## Can I get word alignments while translating?

### Raw alignments from averaging Transformer attention heads

Currently, we support producing word alignment while translating for Transformer based models. Using `-report_align` when calling `translate.py` will output the inferred alignments in Pharaoh format. Those alignments are computed from an argmax on the average of the attention heads of the *second to last* decoder layer. The resulting alignment src-tgt (Pharaoh) will be pasted to the translation sentence, separated by ` ||| `.
Note: The *second to last* default behaviour was empirically determined. It is not the same as the paper (they take the *penultimate* layer), probably because of slight differences in the architecture.

* alignments use the standard "Pharaoh format", where a pair `i-j` indicates the i<sub>th</sub> word of source language is aligned to j<sub>th</sub> word of target language.
* Example: {'src': 'das stimmt nicht !'; 'output': 'that is not true ! ||| 0-0 0-1 1-2 2-3 1-4 1-5 3-6'}
* Using `-tgt` and `-gold_align` options when calling `translate.py`, we output alignments between the source and the gold target rather than the inferred target, assuming we're doing evaluation.
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

Additional word-level information can be incorporated into the model by defining word features in the source sentence. 

Word features must be appended to the actual textual data by using the special character ￨ as a feature separator. For instance:

```
however￨C ￭,￨N according￨L to￨L the￨L logs￨L ￭,￨N she￨L is￨L hard-working￨L ￭.￨N
```

Prior tokenization is not necessary, features will be inferred by using the `FeatInferTransform` transform if tokenization has been applied. For instace:

```
SRC: however,￨C according￨L to￨L the￨L logs,￨L she￨L is￨L hard-working.￨L
TOKENIZED SRC: however ￭, according to the logs ￭, she is hard-working ￭.
RESULT: however￨C ￭,￨C according￨L to￨L the￨L logs￨L ￭,￨L she￨L is￨L hard￨L ￭-￭￨L working￨L ￭.￨L
```

**Options**
- `-n_src_feats`: the expected number of source features per token.
- `-src_feats_defaults` (optional): provides default values for features. This can be really useful when mixing task specific data (with features) with general data which has not been annotated.

For the Transformer architecture make sure the following options are appropriately set:

- `src_word_vec_size` and `tgt_word_vec_size` or `word_vec_size`
- `feat_merge`: how to handle features vecs
- `feat_vec_size` or maybe `feat_vec_exponent`

**Notes**
- `FeatInferTransform` transform is required in order to ensure the functionality.
- Not possible to do shared embeddings (at least with `feat_merge: concat` method)

Sample config file:

```
data:
    dummy:
        path_src: data/train/data.src
        path_tgt: data/train/data.tgt
        transforms: [onmt_tokenize, inferfeats, filtertoolong]
        weight: 1
    valid:
        path_src: data/valid/data.src
        path_tgt: data/valid/data.tgt
        transforms: [onmt_tokenize, inferfeats]

# Transform options
reversible_tokenization: "joiner"

# Vocab opts
src_vocab: exp/data.vocab.src
tgt_vocab: exp/data.vocab.tgt

# Features options
n_src_feats: 2
src_feats_defaults: "0￨1"
feat_merge: "sum"
```

To allow source features in the server add the following parameters in the server's config file:

```
"features": {
    "n_src_feats": 2,
    "src_feats_defaults": "0￨1",
    "reversible_tokenization": "joiner"
}
```

## How can I set up a translation server ?
A REST server was implemented to serve OpenNMT-py models. A discussion is opened on the OpenNMT forum: [discussion link](https://forum.opennmt.net/t/simple-opennmt-py-rest-server/1392).

### I. How it works?

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
    - `report_align`: (bool) return word alignment in pharaoh ('src-tgt') format for every space separated word.
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
