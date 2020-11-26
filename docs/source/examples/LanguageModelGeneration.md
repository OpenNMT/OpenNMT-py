# Language Model Generation


## Step 0: Download and clean the data

Preliminary steps are defined in the [`examples/scripts/prepare_wikitext-103_data.sh`](https://github.com/OpenNMT/OpenNMT-py/tree/master/examples/scripts/prepare_wikitext-103_data.sh). The following command will download the WikiText103 dataset, remove empty lines and shuffle the training corpus:
```bash
chmod u+x prepare_wikitext-103_data.sh
./prepare_wikitext-103_data.sh
```

## Step 1: Prepare the subword model - BPE with pyonmttok

This snippet will train a bpe of 40000 symbols on the train dataset. The bpe model will be stored in "subwords.bpe" and the train, valid and test set will be tokenized and saved.

The tokenized files won't be used for training. Indeed, dynamic iteration over the training dataset allows on the fly tokenization using transforms (see step 2). 

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
An example if a yaml configuration for language modeling task is available in [`examples/wiki_103.yaml`](https://github.com/OpenNMT/OpenNMT-py/tree/master/examples/wiki_103.yaml)

### Language Model specificities

In LM tasks we expect a single source, therefore path_tgt is not required for LM task.

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
The vocabulary is build using:
```bash
onmt_build_vocab -config examples/wiki_103.yaml -n_sample -1
```

## Step 3: Train the model
To train a model for LM tasks, the following parameters must be in the config file.

* *model_task* is used to specify that the task will be language modeling (decoder only model with tansform_lm decoder type, source only dataset expected)
* *decoder_type* must be transform_lm. This transformer is the one used in GPT-2: **Language Models are Unsupervised Multitask Learners**. Basically, it is a transformer without the encoder attention block.
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
Tensorboard can be used to monitor the training

## Step 4: Generate output
Options contained in the loaded model will trigger language modeling inference. When batch mode is used the end of sequences will be repeated in the predictions.

*input.txt* must contain already tokenized, with the same method as the training data. Here, part of validation data will be used:
```bash
head data/wikitext-103-raw/wiki.valid.bpe | cut -d" " -f-15 > data/lm_input.txt
```

To proceed with inference:
```bash
onmt_translate -model data/wikitext-103-raw/run/model-lm_step_1000000.pt -src data/lm_input.txt -output data/lm_pred_input.txt -verbose -n_best 3
```