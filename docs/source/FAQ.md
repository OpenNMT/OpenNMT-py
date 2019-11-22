# FAQ

## How do I use Pretrained embeddings (e.g. GloVe)?

Using vocabularies from OpenNMT-py preprocessing outputs, `embeddings_to_torch.py` to generate encoder and decoder embeddings initialized with GloVe's values.

the script is a slightly modified version of ylhsieh's one2.

Usage:

```shell
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

```shell
mkdir "glove_dir"
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d "glove_dir"
```

2) prepare data:

```shell
onmt_preprocess \
-train_src data/train.src.txt \
-train_tgt data/train.tgt.txt \
-valid_src data/valid.src.txt \
-valid_tgt data/valid.tgt.txt \
-save_data data/data
```

3) prepare embeddings:

```shell
./tools/embeddings_to_torch.py -emb_file_both "glove_dir/glove.6B.100d.txt" \
-dict_file "data/data.vocab.pt" \
-output_file "data/embeddings"
```

4) train using pre-trained embeddings:

```shell
onmt_train -save_model data/model \
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
setup. We have confirmed the following command can replicate their WMT results.

```shell
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
* `label_smoothing 0.1`: use label smoothing loss.

## Do you support multi-gpu?

First you need to make sure you `export CUDA_VISIBLE_DEVICES=0,1,2,3`.

If you want to use GPU id 1 and 3 of your OS, you will need to `export CUDA_VISIBLE_DEVICES=1,3`

Both `-world_size` and `-gpu_ranks` need to be set. E.g. `-world_size 4 -gpu_ranks 0 1 2 3` will use 4 GPU on this node only.

If you want to use 2 nodes with 2 GPU each, you need to set `-master_ip` and `-master_port`, and

* `-world_size 4 -gpu_ranks 0 1`: on the first node
* `-world_size 4 -gpu_ranks 2 3`: on the second node
* `-accum_count 2`: This will accumulate over 2 batches before updating parameters.

if you use a regular network card (1 Gbps) then we suggest to use a higher `-accum_count` to minimize the inter-node communication.

**Note:**

When training on several GPUs, you can't have them in 'Exclusive' compute mode (`nvidia-smi -c 3`).

The multi-gpu setup relies on a Producer/Consumer setup. This setup means there will be `2<n_gpu> + 1` processes spawned, with 2 processes per GPU, one for model training and one (Consumer) that hosts a `Queue` of batches that will be processed next. The additional process is the Producer, creating batches and sending them to the Consumers. This setup is beneficial for both wall time and memory, since it loads data shards 'in advance', and does not require to load it for each GPU process.

## How can I ensemble Models at inference?

You can specify several models in the translate.py command line: -model model1_seed1 model2_seed2
Bear in mind that your models must share the same target vocabulary.

## How can I weight different corpora at training?

### Preprocessing

We introduced `-train_ids` which is a list of IDs that will be given to the preprocessed shards.

E.g. we have two corpora : `parallel.en` and  `parallel.de` + `from_backtranslation.en` `from_backtranslation.de`, we can pass the following in the `preprocess.py` command:

```shell
...
-train_src parallel.en from_backtranslation.en \
-train_tgt parallel.de from_backtranslation.de \
-train_ids A B \
-save_data my_data \
...
```

and it will dump `my_data.train_A.X.pt` based on `parallel.en`//`parallel.de` and `my_data.train_B.X.pt` based on `from_backtranslation.en`//`from_backtranslation.de`.

### Training

We introduced `-data_ids` based on the same principle as above, as well as `-data_weights`, which is the list of the weight each corpus should have.
E.g.

```shell
...
-data my_data \
-data_ids A B \
-data_weights 1 7 \
...
```

will mean that we'll look for `my_data.train_A.*.pt` and `my_data.train_B.*.pt`, and that when building batches, we'll take 1 example from corpus A, then 7 examples from corpus B, and so on.

**Warning**: This means that we'll load as many shards as we have `-data_ids`, in order to produce batches containing data from every corpus. It may be a good idea to reduce the `-shard_size` at preprocessing.

## How do I use BERT?
BERT is a general-purpose "language understanding" model introduced by Google, it can be used for various downstream NLP tasks and easily adapted into a new task using transfer learning. Using BERT has two stages: Pre-training and fine-tuning. But as the Pre-training is super expensive, we do not recommand you to pre-train a BERT from scratch. Instead loading weights from a existing pretrained model and fine-tuning is suggested. Currently we support sentence(-pair) classification and token tagging downstream task.

### Use pretrained BERT weights
To use weights from a existing huggingface's pretrained model, we provide you a script to convert huggingface's BERT model weights into ours.

Usage:
```bash
python bert_ckp_convert.py   --layers NUMBER_LAYER
                             --bert_model_weights_file HUGGINGFACE_BERT_WEIGHTS
                             --output_name OUTPUT_FILE
```
* Go to [modeling_bert.py](https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/modeling_bert.py) to check all available pretrained model.

### Preprocess train/dev dataset
To generate train/dev data for BERT, you can use preprocess_bert.py by providing raw data in certain format and choose a BERT Tokenizer model `-vm` coherent with pretrained model.
#### Classification
For classification dataset, we support input file in csv or plain text file format.

* For csv file, each line should contain a instance with one or two sentence column and one column for label as in GLUE dataset, other csv format dataset should be compatible. A typical csv file should be like:

  | ID | SENTENCE_A               | SENTENCE_B(Optional)      | LABEL   |
  | -- | ------------------------ | ------------------------  | ------- |
  | 0  | sentence a of instance 0 | sentence b of instance 0  | class 2 |
  | 1  | sentence a of instance 1 | sentence b of instance 1  | class 1 |
  | ...| ... | ...  | ... |

  Then calling `preprocess_bert.py` and providing input sentence columns and label column:
  ```bash
  python preprocess_bert.py --task classification --corpus_type {train, valid}
                            --file_type csv [--delimiter '\t'] [--skip_head]
                            --input_columns 1 2 --label_column 3
                            --data DATA_DIR/FILENAME.tsv
                            --save_data dataset
                            -vm bert-base-cased --max_seq_len 256 [--do_lower_case]
                            [--sort_label_vocab] [--do_shuffle]
  ```
* For plain text format, we accept multiply files as input, each file contains instances for one specific class. Each line of the file contains one instance which could be composed by one sentence or two separated by ` ||| `. All input file should be arranged in following way:
  ```
     .
     ├── LABEL_A
     │   └── FILE_WITH_INSTANCE_A
     └── LABEL_B
         └── FILE_WITH_INSTANCE_B
  ```
  Then call `preprocess_bert.py` as following to generate training data:
  ```bash
  python preprocess_bert.py --task classification --corpus_type {'train', 'valid'}
                            --file_type txt [--delimiter ' ||| ']
                            --data DIR_BASE/LABEL_1/FILENAME1 ... DIR_BASE/LABEL_N/FILENAME2
                            --save_data dataset
                            --vocab_model {bert-base-uncased,...}
                            --max_seq_len 256 [--do_lower_case]
                            [--sort_label_vocab] [--do_shuffle]
  ```
#### Tagging
For tagging dataset, we support input file in plain text file format.

Each line of the input file should contain one token and its tagging, different fields should be separated by a delimiter(default space) while sentences are separated by a blank line.

A example of input file is given below (`Token X X Label`):
  ```
  -DOCSTART- -X- O O

  CRICKET NNP I-NP O
  - : O O
  LEICESTERSHIRE NNP I-NP I-ORG
  TAKE NNP I-NP O
  OVER IN I-PP O
  AT NNP I-NP O
  TOP NNP I-NP O
  AFTER NNP I-NP O
  INNINGS NNP I-NP O
  VICTORY NN I-NP O
  . . O O

  LONDON NNP I-NP I-LOC
  1996-08-30 CD I-NP O

  ```
Then call preprocess_bert.py providing token column and label column as following to generate training data for token tagging task:
  ```bash
  python preprocess_bert.py --task tagging --corpus_type {'train', 'valid'}
                            --file_type txt [--delimiter ' ']
                            --input_columns 1 --label_column 3
                            --data DATA_DIR/FILENAME
                            --save_data dataset
                            --vocab_model {bert-base-uncased,...}
                            --max_seq_len 256 [--do_lower_case]
                            [--sort_label_vocab] [--do_shuffle]
  ```
#### Pretraining objective
Even if it's not recommended, we also provide you a script to generate pretraining dataset as you may want to finetuning a existing pretrained model on masked language modeling and next sentence prediction.

The script expects a single file as input, consisting of untokenized text, with one sentence per line, and one blank line between documents.
A usage example is given below:
```bash
python pregenerate_bert_training_data.py  --input_file INPUT_FILE
                                          --output_dir OUTPUT_DIR
                                          --output_name OUTPUT_FILE_PREFIX
                                          --corpus_type {'train', 'valid'}
                                          --vocab_model {bert-base-uncased,...}
                                          [--do_lower_case] [--do_whole_word_mask] [--reduce_memory]
                                          --epochs_to_generate 2
                                          --max_seq_len 128
                                          --short_seq_prob 0.1 --masked_lm_prob 0.15
                                          --max_predictions_per_seq 20
                                          [--save_json]
```

### Training
After preprocessed data have been generated, you can load weights from a pretrained BERT and transfer it to downstream task with a task specific output head. This task specific head will be initialized by a method you choose if there is no such architecture in weights file specified by `--train_from`. Among all available optimizers, you are suggested to use `--optim bertadam` as it is the method used to train BERT. `warmup_steps` could be set as 1% of `train_steps` as in original paper if use linear decay method.

A usage example is given below:
```bash
python train.py  --is_bert --task_type {pretraining, classification, tagging}
                 --data PREPROCESSED_DATAIFILE
                 --train_from CONVERTED_CHECKPOINT.pt [--param_init 0.1]
                 --save_model MODEL_PREFIX --save_checkpoint_steps 1000
                 [--world_size 2] [--gpu_ranks 0 1]
                 --word_vec_size 768 --rnn_size 768
                 --layers 12 --heads 8 --transformer_ff 3072
                 --activation gelu --dropout 0.1 --average_decay 0.0001
                 --batch_size 8 [--accum_count 4] --optim bertadam [--max_grad_norm 0]
                 --learning_rate 2e-5 --learning_rate_decay 0.99 --decay_method linear
                 --train_steps 4000 --valid_steps 200 --warmup_steps 40
                 [--report_every 10] [--seed 3435]
                 [--tensorboard] [--tensorboard_log_dir LOGDIR]
```

### Predicting
After training, you can use `predict.py` to generate predicting for raw file. Make sure to use the same BERT Tokenizer model `--vocab_model` as in training data.

For classification task, file to be predicted should be one sentence(-pair) a line with ` ||| ` separating sentence.
For tagging task, each line should be a tokenized sentence with tokens separated by space.

Usage:
```bash
python predict.py  --task {classification, tagging}
                   --model ONMT_BERT_CHECKPOINT.pt
                   --vocab_model bert-base-uncased [--do_lower_case]
                   --data DATA_2_PREDICT [--delimiter {' ||| ', ' '}] --max_seq_len 256
                   --output PREDICT.txt [--batch_size 8] [--gpu 1] [--seed 3435]
```
## Can I get word alignment while translating?

### Raw alignments from averaging Transformer attention heads

Currently, we support producing word alignment while translating for Transformer based models. Using `-report_align` when calling `translate.py` will output the inferred alignments in Pharaoh format. Those alignments are computed from an argmax on the average of the attention heads of the *second to last* decoder layer. The resulting alignment src-tgt (Pharaoh) will be pasted to the translation sentence, separated by ` ||| `.
Note: The *second to last* default behaviour was empirically determined. It is not the same as the paper (they take the *penultimate* layer), probably because of light differences in the architecture.

* alignments use the standard "Pharaoh format", where a pair `i-j` indicates the i<sub>th</sub> word of source language is aligned to j<sub>th</sub> word of target language.
* Example: {'src': 'das stimmt nicht !'; 'output': 'that is not true ! ||| 0-0 0-1 1-2 2-3 1-4 1-5 3-6'}
* Using the`-tgt` option when calling `translate.py`, we output alignments between the source and the gold target rather than the inferred target, assuming we're doing evaluation.
* To convert subword alignments to word alignments, or symetrize bidirectional alignments, please refer to the [lilt scripts](https://github.com/lilt/alignment-scripts).

### Supervised learning on a specific head

The quality of output alignments can be further improved by providing reference alignments while training. This will invoke multi-task learning on translation and alignment. This is an implementation based on the paper [Jointly Learning to Align and Translate with Transformer Models](https://arxiv.org/abs/1909.02074).

The data need to be preprocessed with the reference alignments in order to learn the supervised task.

When calling `preprocess.py`, add:

* `--train_align <path>`: path(s) to the training alignments in Pharaoh format
* `--valid_align <path>`: path to the validation set alignments in Pharaoh format (optional).
The reference alignment file(s) could be generated by [GIZA++](https://github.com/moses-smt/mgiza/) or [fast_align](https://github.com/clab/fast_align).

Note: There should be no blank lines in the alignment files provided.

Options to learn such alignments are:

* `-lambda_align`: set the value > 0.0 to enable joint align training, the paper suggests 0.05;
* `-alignment_layer`: indicate the index of the decoder layer;
* `-alignment_heads`:  number of alignment heads for the alignment task - should be set to 1 for the supervised task, and preferably kept to default (or same as `num_heads`) for the average task;
* `-full_context_alignment`: do full context decoder pass (no future mask) when computing alignments. This will slow down the training (~12% in terms of tok/s) but will be beneficial to generate better alignment.
