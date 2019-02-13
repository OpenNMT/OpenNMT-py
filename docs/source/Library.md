# Library: Example

For this example, we will assume that we have run preprocess to
create our datasets. For instance

> python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/data -src_vocab_size 1000 -tgt_vocab_size 1000



```python
import torch
import torch.nn as nn

import onmt
import onmt.inputters
import onmt.modules
import onmt.utils
```

We begin by loading in the vocabulary for the model of interest. This will let us check vocab size and to get the special ids for padding.


```python
vocab_fields = torch.load("../../data/data.vocab.pt")

src_text_field = vocab_fields["src"][0][1].base_field
src_vocab = src_text_field.vocab
src_padding = src_vocab.stoi[src_text_field.pad_token]

tgt_text_field = vocab_fields['tgt'][0][1].base_field
tgt_vocab = tgt_text_field.vocab
tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]
```

Next we specify the core model itself. Here we will build a small model with an encoder and an attention based input feeding decoder. Both models will be RNNs and the encoder will be bidirectional


```python
emb_size = 10
rnn_size = 6
# Specify the core model.
encoder_embeddings = onmt.modules.Embeddings(emb_size, len(src_vocab),
                                             word_padding_idx=src_padding)

encoder = onmt.encoders.RNNEncoder(hidden_size=rnn_size, num_layers=1,
                                   rnn_type="LSTM", bidirectional=True,
                                   embeddings=encoder_embeddings)

decoder_embeddings = onmt.modules.Embeddings(emb_size, len(tgt_vocab),
                                             word_padding_idx=tgt_padding)
decoder = onmt.decoders.decoder.InputFeedRNNDecoder(
    hidden_size=rnn_size, num_layers=1, bidirectional_encoder=True, 
    rnn_type="LSTM", embeddings=decoder_embeddings)

model = onmt.models.model.NMTModel(encoder, decoder)

# Specify the tgt word generator and loss computation module
model.generator = nn.Sequential(
    nn.Linear(rnn_size, len(tgt_vocab)),
    nn.LogSoftmax())
builder = onmt.translate.TranslationBuilder(data=valid_data_dataset, fields=vocab_fields)

loss = onmt.utils.loss.NMTLossCompute(
    criterion=nn.NLLLoss(ignore_index=tgt_padding, reduction='sum'),
    generator=model.generator))

```

Now we set up the optimizer. Our wrapper around a core torch optim class handles learning rate updates and gradient normalization automatically.


```python
lr = 1
torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optim = onmt.utils.optimizers.Optimizer(
    torch_optimizer, learning_rate=lr, max_grad_norm=2)
```

Now we load the data from disk with the values of the associated vocab fields. To iterate through the data itself we use a wrapper around a torchtext iterator class. We specify one for both the training and test data.


```python
# Load some data
from itertools import chain
train_data_file = "data/data.train.0.pt"
valid_data_file = "data/data.valid.0.pt"
dataset_fields = dict(chain.from_iterable(vocab_fields.values()))
train_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[train_data_file],
                                                     fields = dataset_fields,
                                                     batch_size=10,
                                                     batch_size_multiple=1,
                                                     batch_size_fn=None,
                                                     device="cpu",
                                                     is_train=True,
                                                     repeat=False)

valid_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[valid_data_file],
                                                     fields=dataset_fields,
                                                     batch_size=10,
                                                     batch_size_multiple=1,
                                                     batch_size_fn=None,
                                                     device="cpu",
                                                     is_train=False,
                                                     repeat=False)
```

Finally we train. Keeping track of the output requires a report manager.


```python
report_mgr = onmt.utils.ReportMgr(
    report_every=1, start_time=None, tensorboard_writer=None)
trainer = onmt.Trainer(model=model,
                       train_loss=loss,
                       valid_loss=loss,
                       optim=optim,
                       report_manager=report_mgr)
trainer.train(train_iter=train_iter,
              train_steps=20,
              valid_iter=valid_iter,
              valid_steps=10)
```

```
[2019-02-13 10:17:03,247 INFO] Start training loop and validate every 10 steps...
[2019-02-13 10:17:03,329 INFO] Loading dataset from data/data.train.0.pt, number of examples: 10000
[2019-02-13 10:17:03,392 INFO] Step  1/   20; acc:   0.00; ppl:  0.98; xent: -0.02; lr: 1.00000; 1046/1310 tok/s;      0 sec
[2019-02-13 10:17:03,460 INFO] Step  2/   20; acc:  33.33; ppl:  0.46; xent: -0.78; lr: 1.00000; 2967/3088 tok/s;      0 sec
[2019-02-13 10:17:03,497 INFO] Step  3/   20; acc:  41.71; ppl:  0.11; xent: -2.24; lr: 1.00000; 5366/4942 tok/s;      0 sec
[2019-02-13 10:17:03,535 INFO] Step  4/   20; acc:  32.34; ppl:  0.05; xent: -3.09; lr: 1.00000; 4372/4563 tok/s;      0 sec
[2019-02-13 10:17:03,598 INFO] Step  5/   20; acc:  33.47; ppl:  0.01; xent: -4.75; lr: 1.00000; 3304/4126 tok/s;      0 sec
[2019-02-13 10:17:03,635 INFO] Step  6/   20; acc:  36.62; ppl:  0.00; xent: -6.94; lr: 1.00000; 3622/3956 tok/s;      0 sec
[2019-02-13 10:17:03,685 INFO] Step  7/   20; acc:  35.00; ppl:  0.00; xent: -8.52; lr: 1.00000; 4075/4075 tok/s;      0 sec
[2019-02-13 10:17:03,721 INFO] Step  8/   20; acc:  37.04; ppl:  0.00; xent: -11.13; lr: 1.00000; 4104/3957 tok/s;      0 sec
[2019-02-13 10:17:03,782 INFO] Step  9/   20; acc:  29.56; ppl:  0.00; xent: -10.46; lr: 1.00000; 4983/4551 tok/s;      1 sec
[2019-02-13 10:17:03,817 INFO] Step 10/   20; acc:  33.60; ppl:  0.00; xent: -13.94; lr: 1.00000; 2128/3746 tok/s;      1 sec
[2019-02-13 10:17:03,852 INFO] Loading dataset from data/data.valid.0.pt, number of examples: 3000
[2019-02-13 10:17:11,890 INFO] Validation perplexity: 9.35574e-08
[2019-02-13 10:17:11,890 INFO] Validation accuracy: 35.5637
[2019-02-13 10:17:11,939 INFO] Step 11/   20; acc:  31.22; ppl:  0.00; xent: -14.34; lr: 1.00000;  22/ 27 tok/s;      9 sec
[2019-02-13 10:17:11,977 INFO] Step 12/   20; acc:  29.45; ppl:  0.00; xent: -15.54; lr: 1.00000; 3367/4573 tok/s;      9 sec
[2019-02-13 10:17:12,043 INFO] Step 13/   20; acc:  32.09; ppl:  0.00; xent: -18.27; lr: 1.00000; 3233/4087 tok/s;      9 sec
[2019-02-13 10:17:12,141 INFO] Step 14/   20; acc:  32.49; ppl:  0.00; xent: -19.98; lr: 1.00000; 4844/4104 tok/s;      9 sec
[2019-02-13 10:17:12,186 INFO] Step 15/   20; acc:  32.50; ppl:  0.00; xent: -22.22; lr: 1.00000; 5148/5616 tok/s;      9 sec
[2019-02-13 10:17:12,261 INFO] Step 16/   20; acc:  41.39; ppl:  0.00; xent: -29.82; lr: 1.00000; 4020/3291 tok/s;      9 sec
[2019-02-13 10:17:12,342 INFO] Step 17/   20; acc:  35.77; ppl:  0.00; xent: -27.76; lr: 1.00000; 3601/3462 tok/s;      9 sec
[2019-02-13 10:17:12,360 INFO] Step 18/   20; acc:  25.00; ppl:  0.00; xent: -22.50; lr: 1.00000; 3989/4594 tok/s;      9 sec
[2019-02-13 10:17:12,444 INFO] Step 19/   20; acc:  33.77; ppl:  0.00; xent: -29.69; lr: 1.00000; 4752/4606 tok/s;      9 sec
[2019-02-13 10:17:12,491 INFO] Step 20/   20; acc:  32.72; ppl:  0.00; xent: -31.11; lr: 1.00000; 4915/4848 tok/s;      9 sec
[2019-02-13 10:17:12,530 INFO] Loading dataset from data/data.valid.0.pt, number of examples: 3000
[2019-02-13 10:17:20,607 INFO] Validation perplexity: 4.92281e-16
[2019-02-13 10:17:20,608 INFO] Validation accuracy: 35.5637
```

To use the model, we need to load up the translation functions. A Translator object requires the vocab fields, a global scorer, general options and model options, here default values collected with dummy parsers.


```python
import onmt.translate
import configargparse

dummy_parser = configargparse.ArgumentParser()
onmt.opts.model_opts(dummy_parser)
model_opt = dummy_parser.parse_known_args([])[0]

dummy_parser = configargparse.ArgumentParser()
onmt.opts.translate_opts(dummy_parser)
opt = dummy_parser.parse_args("-model dummymodel -src dummysrc")

scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)
translator = onmt.translate.Translator(model=model,
                                       fields=vocab_fields,
                                       opt=opt,
                                       model_opt=model_opt,
                                       global_scorer=scorer)

builder = onmt.translate.TranslationBuilder(data=valid_data_dataset,
                                            fields=vocab_fields)

for batch in valid_iter:
    trans_batch = translator.translate_batch(
        batch=batch, src_vocabs=valid_data_dataset.src_vocabs,
        attn_debug=False)
    translations = builder.from_batch(trans_batch)
    for trans in translations:
        print(trans.log(0))
    break
```
```
[2019-02-13 10:19:18,802 INFO] Loading dataset from data/data.valid.0.pt, number of examples: 3000


SENT 0: ['Parliament', 'Does', 'Not', 'Support', 'Amendment', 'Freeing', 'Tymoshenko']
PRED 0: <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
PRED SCORE: 9153.7441


SENT 0: ['Today', ',', 'the', 'Ukraine', 'parliament', 'dismissed', ',', 'within', 'the', 'Code', 'of', 'Criminal', 'Procedure', 'amendment', ',', 'the', 'motion', 'to', 'revoke', 'an', 'article', 'based', 'on', 'which', 'the', 'opposition', 'leader', ',', 'Yulia', 'Tymoshenko', ',', 'was', 'sentenced', '.']
PRED 0: <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
PRED SCORE: 9153.3516


SENT 0: ['The', 'amendment', 'that', 'would', 'lead', 'to', 'freeing', 'the', 'imprisoned', 'former', 'Prime', 'Minister', 'was', 'revoked', 'during', 'second', 'reading', 'of', 'the', 'proposal', 'for', 'mitigation', 'of', 'sentences', 'for', 'economic', 'offences', '.']
PRED 0: <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
PRED SCORE: 9153.1836


SENT 0: ['In', 'October', ',', 'Tymoshenko', 'was', 'sentenced', 'to', 'seven', 'years', 'in', 'prison', 'for', 'entering', 'into', 'what', 'was', 'reported', 'to', 'be', 'a', 'disadvantageous', 'gas', 'deal', 'with', 'Russia', '.']
PRED 0: <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
PRED SCORE: 9153.3281


SENT 0: ['The', 'verdict', 'is', 'not', 'yet', 'final;', 'the', 'court', 'will', 'hear', 'Tymoshenko', '&apos;s', 'appeal', 'in', 'December', '.']
PRED 0: <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
PRED SCORE: 9153.2695


SENT 0: ['Tymoshenko', 'claims', 'the', 'verdict', 'is', 'a', 'political', 'revenge', 'of', 'the', 'regime;', 'in', 'the', 'West', ',', 'the', 'trial', 'has', 'also', 'evoked', 'suspicion', 'of', 'being', 'biased', '.']
PRED 0: <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
PRED SCORE: 9153.6104


SENT 0: ['The', 'proposal', 'to', 'remove', 'Article', '365', 'from', 'the', 'Code', 'of', 'Criminal', 'Procedure', ',', 'upon', 'which', 'the', 'former', 'Prime', 'Minister', 'was', 'sentenced', ',', 'was', 'supported', 'by', '147', 'members', 'of', 'parliament', '.']
PRED 0: <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
PRED SCORE: 9153.1816


SENT 0: ['Its', 'ratification', 'would', 'require', '226', 'votes', '.']
PRED 0: <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
PRED SCORE: 9153.6611


SENT 0: ['Libya', '&apos;s', 'Victory']
PRED 0: <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
PRED SCORE: 9149.8838


SENT 0: ['The', 'story', 'of', 'Libya', '&apos;s', 'liberation', ',', 'or', 'rebellion', ',', 'already', 'has', 'its', 'defeated', '.']
PRED 0: <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
PRED SCORE: 9153.0166
