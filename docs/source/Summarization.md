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
For CNNDM we follow See et al. [2] and additionally truncate the source length at 400 tokens and the target at 100.

**command used**:

(1) CNNDM

```
python preprocess.py -train_src data/cnndm/train.txt.src -train_tgt data/cnn-no-sent-tag/train.txt.tgt -valid_src data/cnndm/val.txt.src -valid_tgt data/cnn-no-sent-tag/val.txt.tgt -save_data data/cnn-no-sent-tag/cnndm -src_seq_length 10000 -tgt_seq_length 10000 -src_seq_length_trunc 400 -tgt_seq_length_trunc 100 -dynamic_dict -share_vocab
```

(2) Gigaword

```
python preprocess.py -train_src data/giga/train.article.txt -train_tgt data/giga/train.title.txt -valid_src data/giga/valid.article.txt -valid_tgt data/giga/valid.title.txt -save_data data/giga/giga -src_seq_length 10000 -dynamic_dict -share_vocab
```


### Training

The training procedure described in this section for the most part follows parameter choices and implementation similar to that of See et al. [2].  As mentioned above, we use copy attention as a mechanism for the model to decide whether to either generate a new word or to copy from the source (`copy_attn`).
A notable difference to See's model is that we are using the attention mechanism introduced by Bahdanau et al. [3] (`global_attention mlp`) instead of that by Luong et al. [4] (`global_attention dot`). Both options typically perform very similar to each other with Luong attention often having a slight advantage.
We are using using a 128-dimensional word-embedding, and 512-dimensional 1 layer LSTM. On the encoder side, we use a bidirectional LSTM (`brnn`), which means that the 512 dimensions are split into 256 dimensions per direction.
We also share the word embeddings between encoder and decoder (`share_embeddings`). This option drastically reduces the number of parameters the model has to learn. However, we found only minimal impact on performance of a model without this option.

For the training procedure, we are using SGD with an initial learning rate of 1 for a total of 16 epochs. In most cases, the lowest validation perplexity is achieved around epoch 10-12. We also use OpenNMT's default learning rate decay, which halves the learning rate after every epoch once the validation perplexity increased after an epoch (or after epoch 8).
Alternative training procedures such as adam with initial learning rate 0.001 converge faster than sgd, but achieve slightly worse. We additionally set the maximum norm of the gradient to 2, and renormalize if the gradient norm exceeds this value.

**commands used**:

(1) CNNDM

```
python train.py -save_model logs/notag_sgd3 -data data/cnn-no-sent-tag/CNNDM -copy_attn -global_attention mlp -word_vec_size 128 -rnn_size 256 -layers 1 -brnn -epochs 16 -seed 777 -batch_size 32 -max_grad_norm 2 -share_embeddings -gpuid 0 -start_checkpoint_at 9
```

(2) Gigaword

```
python train.py -save_model logs/giga_sgd3_512 -data data/giga/giga -copy_attn -global_attention mlp -word_vec_size 128 -rnn_size 512 -layers 1 -brnn -epochs 16 -seed 777 -batch_size 32 -max_grad_norm 2 -share_embeddings -gpuid 0 -start_checkpoint_at 9
```


### Inference

During inference, we use beam-search with a beam-size of 10.
We additionally use the `replace_unk` option which replaces generated `<UNK>` tokens with the source token of highest attention. This acts as safety-net should the copy attention fail which should learn to copy such words.

**commands used**:

(1) CNNDM

```
python translate.py -gpu 2 -batch_size 1 -model logs/notag_try3_acc_49.29_ppl_14.62_e16.pt -src data/cnndm/test.txt.src -output sgd3_out.txt -beam_size 10 -replace_unk
```


(2) Gigaword

```
python translate.py -gpu 2 -batch_size 1 -model logs/giga_sgd3_512_acc_51.10_ppl_12.04_e16.pt -src data/giga/test.article.txt -output giga_sgd3.out.txt -beam_size 10 -replace_unk
```


### Evaluation

#### CNNDM

To evaluate the ROUGE scores on CNNDM, we extended the pyrouge wrapper with additional evaluations such as the amount of repeated n-grams (typically found in models with copy attention), found [here](https://github.com/falcondai/pyrouge/).

It can be run with the following command:

```
python baseline.py -s sgd3_out.txt -t ~/datasets/cnn-dailymail/sent-tagged/test.txt.tgt -m no_sent_tag -r
```

Note that the `no_sent_tag` option strips tags around sentences - when a sentence previously was `<s> w w w w . </s>`, it becomes `w w w w .`.

#### Gigaword

For evaluation of large test sets such as Gigaword, we use the a parallel python wrapper around ROUGE, found [here](https://github.com/pltrdy/files2rouge).

**command used**:
`files2rouge giga_sgd3.out.txt test.title.txt --verbose`

Running the commands above should yield the following scores:
```
ROUGE-1 (F): 0.352127
ROUGE-2 (F): 0.173109
ROUGE-3 (F): 0.098244
ROUGE-L (F): 0.327742
ROUGE-S4 (F): 0.155524
```



### References

[1] Vinyals, O., Fortunato, M. and Jaitly, N., 2015. Pointer Network. NIPS

[2] See, A., Liu, P.J. and Manning, C.D., 2017. Get To The Point: Summarization with Pointer-Generator Networks. ACL

[3] Bahdanau, D., Cho, K. and Bengio, Y., 2014. Neural machine translation by jointly learning to align and translate. ICLR

[4] Luong, M.T., Pham, H. and Manning, C.D., 2015. Effective approaches to attention-based neural machine translation. EMNLP
