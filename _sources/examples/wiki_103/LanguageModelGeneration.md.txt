# Language Model Wiki-103


## Step 0: Download and clean the data

Preliminary steps are defined in the [`docs/source/examples/wiki_103/prepare_wikitext-103_data.sh`](https://github.com/OpenNMT/OpenNMT-py/tree/master/docs/sources/examples/wiki_103/prepare_wikitext-103_data.sh). The following command will download the [WikiText103 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/), remove empty lines and shuffle the training corpus:
```bash
chmod u+x prepare_wikitext-103_data.sh
./prepare_wikitext-103_data.sh
```

## Step 1: Prepare the subword model - BPE with pyonmttok

This snippet will train a bpe of 40000 symbols on the training dataset using pyonmttok. The bpe model will be stored in `subwords.bpe` and the train/valid/test sets will be tokenized and saved.

The tokenized files won't be used for training. Indeed, dynamic iteration over the training dataset enables on the fly tokenization using transforms (see step 2).

```python
import pyonmttok

args = {
    "mode": "aggressive",
    "joiner_annotate": True,
    "preserve_placeholders": True,
    "case_markup": True,
    "soft_case_regions": True,
    "preserve_segmented_tokens": True,
}
n_symbols = 40000

tokenizer_default = pyonmttok.Tokenizer(**args)
learner = pyonmttok.BPELearner(tokenizer=tokenizer_default, symbols=n_symbols)
# load training corpus
learner.ingest_file("wiki.train.raw")

# learn and store bpe model
tokenizer = learner.learn("subwords.bpe")

# tokenize corpus and save results
for data_file in ["wiki.valid", "wiki.test", "wiki.train"]:
    tokenizer.tokenize_file(f"{data_file}.raw", f"{data_file}.bpe")
```

## Step 2: Build the vocabulary
An example of yaml configuration for language modeling task is available in [`examples/wiki_103.yaml`](https://github.com/OpenNMT/OpenNMT-py/tree/master/docs/source/examples/wiki_103/wiki_103.yaml). This configuration will be used for building the vocabulary and training the model.
BPE and language modeling specificities are explained in the following sections.

### Language Model specificities

In LM tasks we expect a single source, therefore path_tgt is not required for LM tasks.

```yaml
data:
    corpus_1:
        path_src: data/wikitext-103-raw/wiki.train.raw
```

### BPE specificities

To use BPE tokenization on the fly, the following parameters must be in the config file.
Slight differences between on the fly tokenization and outputed tokenized files from step 1 can be observed.

```yaml
src_subword_type: bpe
src_subword_model: data/wikitext-103-raw/subwords.bpe
src_onmttok_kwargs: '{"mode": "aggressive", "joiner_annotate": True, "preserve_placeholders":
  True, "case_markup": True, "soft_case_regions": True, "preserve_segmented_tokens":
  True}'
transforms: [onmt_tokenize]
```

### Build vocabulary command
The vocabulary is built using:
```bash
onmt_build_vocab -config examples/wiki_103.yaml -n_sample -1
```

## Step 3: Train the model
To train a model for LM tasks, the following parameters are required:

* *model_task* is used to specify that the task will be language modeling (decoder only model with tansformer_lm decoder type, source only dataset expected)
* *decoder_type* must be transformer_lm. This transformer is the one used in GPT-2: [**Language Models are Unsupervised Multitask Learners**](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Basically, it is a transformer without an encoder attention block
* *encoder_type* is not useful but need to be mentionned
* *share_vocab* must be true. The slided source will play the role of the target hence vocabulary must be shared. 
```yaml
model_task: lm
encoder_type: transformer_lm
decoder_type: transformer_lm

share_vocab: true
```

The training is launched using:
```bash
onmt_train -config examples/wiki_103.yaml
```
Tensorboard can be used to monitor the training.

**Expected results:** perplexity of 20-22 on the validation set.

## Step 4: Generate output
Options contained in the loaded model will trigger language modeling specific inference.

`input.txt` must contain already tokenized examples, with the same method as the training data. Here, part of validation data will be used:
```bash
head data/wikitext-103-raw/wiki.valid.bpe | cut -d" " -f-15 > data/wikitext-103-raw/lm_input.txt
```

To proceed with LM inference, sampling methods such as top-k sampling or nucleus sampling are usually applied. Details and options about inference methods can be found in [`onmt/opts.py`](https://github.com/OpenNMT/OpenNMT-py/tree/master/onmt/opts.py).

The following command will provide inference with nucleus sampling of p=0.9 and return the 3 sequences with the lowest perplexity out of the 10 generated sequences:
```bash
onmt_translate -model data/wikitext-103-raw/run/model-lm_step_1000000.pt -src data/wikitext-103-raw/lm_input.txt -output data/wikitext-103-raw/lm_pred_input.txt -verbose -n_best 3 -random_sampling_topp 0.9 -beam_size 10
```
