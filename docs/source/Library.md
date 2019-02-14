# Library: Example

For this example, we will assume that we have run preprocess to
create our datasets. For instance

> python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/data -src_vocab_size 10000 -tgt_vocab_size 10000



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
vocab_fields = torch.load("data/data.vocab.pt")

src_text_field = vocab_fields["src"][0][1].base_field
src_vocab = src_text_field.vocab
src_padding = src_vocab.stoi[src_text_field.pad_token]

tgt_text_field = vocab_fields['tgt'][0][1].base_field
tgt_vocab = tgt_text_field.vocab
tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]
```

Next we specify the core model itself. Here we will build a small model with an encoder and an attention based input feeding decoder. Both models will be RNNs and the encoder will be bidirectional


```python
emb_size = 100
rnn_size = 500
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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = onmt.models.model.NMTModel(encoder, decoder)
model.to(device)

# Specify the tgt word generator and loss computation module
model.generator = nn.Sequential(
    nn.Linear(rnn_size, len(tgt_vocab)),
    nn.LogSoftmax(dim=-1))

loss = onmt.utils.loss.NMTLossCompute(
    criterion=nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"),
    generator=model.generator)
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
vocab_fields_formatted_for_iterator = dict(chain.from_iterable(vocab_fields.values()))
train_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[train_data_file],
                                                     fields=vocab_fields_formatted_for_iterator,
                                                     batch_size=50,
                                                     batch_size_multiple=1,
                                                     batch_size_fn=None,
                                                     device=device,
                                                     is_train=True,
                                                     repeat=True)

valid_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[valid_data_file],
                                                     fields=vocab_fields_formatted_for_iterator,
                                                     batch_size=10,
                                                     batch_size_multiple=1,
                                                     batch_size_fn=None,
                                                     device=device,
                                                     is_train=False,
                                                     repeat=False)
```

Finally we train. Keeping track of the output requires a report manager.


```python
report_manager = onmt.utils.ReportMgr(
    report_every=50, start_time=None, tensorboard_writer=None)
trainer = onmt.Trainer(model=model,
                       train_loss=loss,
                       valid_loss=loss,
                       optim=optim,
                       report_manager=report_manager)
trainer.train(train_iter=train_iter,
              train_steps=400,
              valid_iter=valid_iter,
              valid_steps=200)
```

```
[2019-02-14 17:40:32,854 INFO] Start training loop and validate every 200 steps...
[2019-02-14 17:40:32,975 INFO] Loading dataset from data/data.train.0.pt, number of examples: 10000
[2019-02-14 17:40:53,996 INFO] Step 50/  400; acc:  12.04; ppl:  0.00; xent: -144.42; lr: 1.00000; 524/517 tok/s;     21 sec
[2019-02-14 17:41:17,655 INFO] Step 100/  400; acc:  11.76; ppl:  0.00; xent: -450.43; lr: 1.00000; 478/481 tok/s;     45 sec
[2019-02-14 17:41:38,788 INFO] Step 150/  400; acc:  11.81; ppl:  0.00; xent: -765.66; lr: 1.00000; 528/517 tok/s;     66 sec
[2019-02-14 17:42:02,082 INFO] Step 200/  400; acc:  12.49; ppl:  0.00; xent: -1107.68; lr: 1.00000; 479/487 tok/s;     89 sec
[2019-02-14 17:42:02,114 INFO] Loading dataset from data/data.valid.0.pt, number of examples: 3000
[2019-02-14 17:42:51,292 INFO] Validation perplexity: 0
[2019-02-14 17:42:51,293 INFO] Validation accuracy: 19.8323
[2019-02-14 17:43:12,535 INFO] Step 250/  400; acc:  12.10; ppl:  0.00; xent: -1401.45; lr: 1.00000; 156/153 tok/s;    160 sec
[2019-02-14 17:43:35,407 INFO] Step 300/  400; acc:  12.60; ppl:  0.00; xent: -1762.64; lr: 1.00000; 497/491 tok/s;    183 sec
[2019-02-14 17:43:56,475 INFO] Step 350/  400; acc:  11.71; ppl:  0.00; xent: -2012.42; lr: 1.00000; 492/506 tok/s;    204 sec
[2019-02-14 17:44:21,052 INFO] Step 400/  400; acc:  12.90; ppl:  0.00; xent: -2433.33; lr: 1.00000; 493/487 tok/s;    228 sec
[2019-02-14 17:44:21,084 INFO] Loading dataset from data/data.valid.0.pt, number of examples: 3000
[2019-02-14 17:45:11,367 INFO] Validation perplexity: 0
[2019-02-14 17:45:11,368 INFO] Validation accuracy: 19.8323
```

To use the model, we need to load up the translation functions. A Translator object requires the vocab fields, readers for source and target and a global scorer.


```python
import onmt.translate

src_reader = onmt.inputters.str2reader["text"]
tgt_reader = onmt.inputters.str2reader["text"]
scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7, 
                                         beta=0., 
                                         length_penalty="avg", 
                                         coverage_penalty="none")
gpu = 0 if torch.cuda.is_available() else -1
translator = onmt.translate.Translator(model=model, 
                                       fields=vocab_fields, 
                                       src_reader=src_reader, 
                                       tgt_reader=tgt_reader, 
                                       global_scorer=scorer,
                                       gpu=gpu)
builder = onmt.translate.TranslationBuilder(data=torch.load(valid_data_file), 
                                            fields=vocab_fields)


for batch in valid_iter:
    trans_batch = translator.translate_batch(
        batch=batch, src_vocabs=[src_vocab],
        attn_debug=False)
    translations = builder.from_batch(trans_batch)
    for trans in translations:
        print(trans.log(0))
```
```
[2019-02-14 17:49:24,398 INFO] Loading dataset from data/data.valid.0.pt, number of examples: 3000


SENT 0: ['Parliament', 'Does', 'Not', 'Support', 'Amendment', 'Freeing', 'Tymoshenko']
PRED 0: <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
PRED SCORE: 11987.6055


SENT 0: ['Today', ',', 'the', 'Ukraine', 'parliament', 'dismissed', ',', 'within', 'the', 'Code', 'of', 'Criminal', 'Procedure', 'amendment', ',', 'the', 'motion', 'to', 'revoke', 'an', 'article', 'based', 'on', 'which', 'the', 'opposition', 'leader', ',', 'Yulia', 'Tymoshenko', ',', 'was', 'sentenced', '.']
PRED 0: <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
PRED SCORE: 11987.6113


SENT 0: ['The', 'amendment', 'that', 'would', 'lead', 'to', 'freeing', 'the', 'imprisoned', 'former', 'Prime', 'Minister', 'was', 'revoked', 'during', 'second', 'reading', 'of', 'the', 'proposal', 'for', 'mitigation', 'of', 'sentences', 'for', 'economic', 'offences', '.']
PRED 0: <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
PRED SCORE: 11987.6104

...
