# Example: Summarization

This document describes how to replicate summarization experiments on the CNNDM and gigaword datasets using OpenNMT-py.
In the following, we assume access to a tokenized form of the corpus split into train/valid/test set.

An example article-title pair from Gigaword should look like this:

**Input**
*australia 's current account deficit shrunk by a record #.## billion dollars -lrb- #.## billion us -rrb- in the june quarter due to soaring commodity prices , figures released monday showed .*

**Output**
*australian current account deficit narrows sharply*


### Preprocessing the data

Since we are using copy-attention [1] in the model, we need to preprocess the dataset such that source and target are aligned and use the same dictionary. This is achieved by using the options `dynamic_dict` and `share_vocab`.
We additionally turn off truncation of the source to ensure that inputs longer than 50 words are not truncated.
For CNNDM we follow See et al. [2] and additionally truncate the source length at 400 tokens and the target at 100. We also note that in CNNDM, we found models to work better if the target surrounds sentences with tags such that a sentence looks like `<t> w1 w2 w3 . </t>`. If you use this formatting, you can remove the tags after the inference step with the commands `sed -i 's/ <\/t>//g' FILE.txt` and `sed -i 's/<t> //g' FILE.txt`.

**Command used**:

(1) CNNDM

```
python preprocess.py -train_src data/cnndm/train.txt.src \
                     -train_tgt data/cnndm/train.txt.tgt \
                     -valid_src data/cnndm/val.txt.src \
                     -valid_tgt data/cnndm/val.txt.tgt \
                     -save_data data/cnndm/CNNDM \
                     -src_seq_length 10000 \
                     -tgt_seq_length 10000 \
                     -src_seq_length_trunc 400 \
                     -tgt_seq_length_trunc 100 \
                     -dynamic_dict \
                     -share_vocab \
                     -max_shard_size (500 * 1024 * 1024)
```

(2) Gigaword

```
python preprocess.py -train_src data/giga/train.article.txt \
                     -train_tgt data/giga/train.title.txt \
                     -valid_src data/giga/valid.article.txt \
                     -valid_tgt data/giga/valid.title.txt \
                     -save_data data/giga/GIGA \
                     -src_seq_length 10000 \
                     -dynamic_dict \
                     -share_vocab \
                     -max_shard_size (500 * 1024 * 1024)
```


### Training

The training procedure described in this section for the most part follows parameter choices and implementation similar to that of See et al. [2]. We describe notable options in the following list:

- `copy_attn`: This is the most important option, since it allows the model to copy words from the source.
- `global_attention mlp`: This makes the model use the  attention mechanism introduced by Bahdanau et al. [3] instead of that by Luong et al. [4] (`global_attention dot`).
- `share_embeddings`: This shares the word embeddings between encoder and decoder. This option drastically decreases the number of parameters a model has to learn. We did not find this option to helpful, but you can try it out by adding it to the command below.
-  `reuse_copy_attn`: This option reuses the standard attention as copy attention. Without this, the model learns an additional attention that is only used for copying.
-  `copy_loss_by_seqlength`: This modifies the loss to divide the loss of a sequence by the number of tokens in it. In practice, we found this to generate longer sequences during inference. However, this effect can also be achieved by using penalties during decoding.
-  `bridge`: This is an additional layer that uses the final hidden state of the encoder as input and computes an initial hidden state for the decoder. Without this, the decoder is initialized with the final hidden state of the encoder directly.
-  `optim adagrad`: Adagrad outperforms SGD when coupled with the following option.
-  `adagrad_accumulator_init 0.1`: PyTorch does not initialize the accumulator in adagrad with any values. To match the optimization algorithm with the Tensorflow version, this option needs to be added.


We are using using a 128-dimensional word-embedding, and 512-dimensional 1 layer LSTM. On the encoder side, we use a bidirectional LSTM (`brnn`), which means that the 512 dimensions are split into 256 dimensions per direction.
We also use OpenNMT's default learning rate decay, which halves the learning rate after every epoch once the validation perplexity increased after an epoch (or after epoch 8).
We additionally set the maximum norm of the gradient to 2, and renormalize if the gradient norm exceeds this value and do not use any dropout.

**commands used**:

(1) CNNDM

```
python train.py -save_model models/cnndm \
                -data data/cnndm/CNNDM \
                -copy_attn \
                -global_attention mlp \
                -word_vec_size 128 \
                -rnn_size 512 \
                -layers 1 \
                -encoder_type brnn \
                -epochs 20 \
                -max_grad_norm 2 \
                -dropout 0. \
                -batch_size 16 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -reuse_copy_attn \
                -copy_loss_by_seqlength \
                -bridge \
                -seed 777 \
                -gpuid X
```

Gigaword can be trained equivalently.


### Inference

During inference, we use beam-search with a beam-size of 5. We also added specific penalties that we can use during decoding, described in the following.

- `stepwise_penalty`:
- `coverage_penalty summary`
- `beta 5`
- `length_penalty wu`
- `alpha 0.8`

**commands used**:

(1) CNNDM

```
python translate.py -gpu X \
                    -batch_size 20 \
                    -beam_size 5 \
                    -model models/cnndm... \
                    -src data/cnndm/test.txt.src \
                    -output testout/cnndm.out \
                    -min_length 35 \
                    -verbose \
                    -stepwise_penalty \
                    -coverage_penalty summary \
                    -beta 5 \
                    -length_penalty wu \
                    -alpha 0.9 \
                    -verbose
```




### Evaluation

#### CNNDM

To evaluate the ROUGE scores on CNNDM, we extended the pyrouge wrapper with additional evaluations such as the amount of repeated n-grams (typically found in models with copy attention), found [here](https://github.com/sebastianGehrmann/rouge-baselines). The repository includes a sub-repo called pyrouge. Make sure to clone the code with the `git clone --recurse-submodules https://github.com/sebastianGehrmann/rouge-baselines` command to check this out as well and follow the installation instructions on the pyrouge repository before calling this script.
The installation instructions can be found [here](https://github.com/falcondai/pyrouge/tree/9cdbfbda8b8d96e7c2646ffd048743ddcf417ed9#installation). Note that on MacOS, we found that the pointer to your perl installation in line 1 of `pyrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl` might be different from the one you have installed. A simple fix is to change this line to `#!/usr/local/bin/perl -w` if it fails.

It can be run with the following command:

```
python baseline.py -s testout/cnndm.out -t data/cnndm/test.txt.tgt -m no_sent_tag -r
```

The `no_sent_tag` option strips tags around sentences - when a sentence previously was `<s> w w w w . </s>`, it becomes `w w w w .`.

#### Gigaword

For evaluation of large test sets such as Gigaword, we use the a parallel python wrapper around ROUGE, found [here](https://github.com/pltrdy/files2rouge).

**command used**:
`files2rouge giga.out test.title.txt --verbose`

### Scores and Models

#### CNNDM

| Model Type    | Model    | R1 R  | R1 P  | R1 F  | R2 R  | R2 P  | R2 F  | RL R  | RL P  | RL F  |
| ------------- |  -------- | -----:| -----:| -----:|------:| -----:| -----:|-----: | -----:| -----:|
| Pointer-Generator + Coverage [2]     | [link](https://github.com/abisee/pointer-generator)     | 39.05 |	43.02 |	39.53 |	17.16 | 18.77 | 17.28  | 35.98 | 39.56 | 36.38 |
| Pointer-Generator [2]  |  [link](https://github.com/abisee/pointer-generator)     | 37.76 | 37.60| 36.44| 16.31| 16.12| 15.66| 34.66| 34.46| 33.42 |
| OpenNMT BRNN      |  link     | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 |
| col 2 is      |  link     | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 |
| zebra stripes |  link     | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 | 33.33 |




### References

[1] Vinyals, O., Fortunato, M. and Jaitly, N., 2015. Pointer Network. NIPS

[2] See, A., Liu, P.J. and Manning, C.D., 2017. Get To The Point: Summarization with Pointer-Generator Networks. ACL

[3] Bahdanau, D., Cho, K. and Bengio, Y., 2014. Neural machine translation by jointly learning to align and translate. ICLR

[4] Luong, M.T., Pham, H. and Manning, C.D., 2015. Effective approaches to attention-based neural machine translation. EMNLP