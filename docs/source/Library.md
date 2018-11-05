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
vocab = dict(torch.load("../../data/data.vocab.pt"))
src_padding = vocab["src"].stoi[onmt.inputters.PAD_WORD]
tgt_padding = vocab["tgt"].stoi[onmt.inputters.PAD_WORD]
```

Next we specify the core model itself. Here we will build a small model with an encoder and an attention based input feeding decoder. Both models will be RNNs and the encoder will be bidirectional


```python
emb_size = 10
rnn_size = 6
# Specify the core model.
encoder_embeddings = onmt.modules.Embeddings(emb_size, len(vocab["src"]),
                                             word_padding_idx=src_padding)

encoder = onmt.encoders.RNNEncoder(hidden_size=rnn_size, num_layers=1,
                                 rnn_type="LSTM", bidirectional=True,
                                 embeddings=encoder_embeddings)

decoder_embeddings = onmt.modules.Embeddings(emb_size, len(vocab["tgt"]),
                                             word_padding_idx=tgt_padding)
decoder = onmt.decoders.decoder.InputFeedRNNDecoder(hidden_size=rnn_size, num_layers=1,
                                           bidirectional_encoder=True,
                                           rnn_type="LSTM", embeddings=decoder_embeddings)
model = onmt.models.model.NMTModel(encoder, decoder)

# Specify the tgt word generator and loss computation module
model.generator = nn.Sequential(
            nn.Linear(rnn_size, len(vocab["tgt"])),
            nn.LogSoftmax())
loss = onmt.utils.loss.NMTLossCompute(model.generator, vocab["tgt"])
```

Now we set up the optimizer. This could be a core torch optim class, or our wrapper which handles learning rate updates and gradient normalization automatically.


```python
optim = onmt.utils.optimizers.Optimizer(method="sgd", learning_rate=1, max_grad_norm=2)
optim.set_parameters(model.named_parameters())
```

Now we load the data from disk. Currently will need to call a function to load the fields into the data as well.


```python
# Load some data
data = torch.load("../../data/data.train.1.pt")
valid_data = torch.load("../../data/data.valid.1.pt")
data.load_fields(vocab)
valid_data.load_fields(vocab)
data.examples = data.examples[:100]
```

To iterate through the data itself we use a torchtext iterator class. We specify one for both the training and test data.


```python
train_iter = onmt.inputters.OrderedIterator(
                dataset=data, batch_size=10,
                device=-1,
                repeat=False)
valid_iter = onmt.inputters.OrderedIterator(
                dataset=valid_data, batch_size=10,
                device=-1,
                train=False)
```

Finally we train.


```python
trainer = onmt.Trainer(model, loss, loss, optim)

def report_func(*args):
    stats = args[-1]
    stats.output(args[0], args[1], 10, 0)
    return stats

for epoch in range(2):
    trainer.train(epoch, report_func)
    val_stats = trainer.validate()

    print("Validation")
    val_stats.output(epoch, 11, 10, 0)
    trainer.epoch_step(val_stats.ppl(), epoch)
```

    Epoch  0,     0/   10; acc:   0.00; ppl: 1225.23; 1320 src tok/s; 1320 tgt tok/s; 1514090454 s elapsed
    Epoch  0,     1/   10; acc:   9.50; ppl: 996.33; 1188 src tok/s; 1194 tgt tok/s; 1514090454 s elapsed
    Epoch  0,     2/   10; acc:  16.51; ppl: 694.48; 1265 src tok/s; 1267 tgt tok/s; 1514090454 s elapsed
    Epoch  0,     3/   10; acc:  20.49; ppl: 470.39; 1459 src tok/s; 1420 tgt tok/s; 1514090454 s elapsed
    Epoch  0,     4/   10; acc:  22.68; ppl: 387.03; 1511 src tok/s; 1462 tgt tok/s; 1514090454 s elapsed
    Epoch  0,     5/   10; acc:  24.58; ppl: 345.44; 1625 src tok/s; 1509 tgt tok/s; 1514090454 s elapsed
    Epoch  0,     6/   10; acc:  25.37; ppl: 314.39; 1586 src tok/s; 1493 tgt tok/s; 1514090454 s elapsed
    Epoch  0,     7/   10; acc:  26.14; ppl: 291.15; 1593 src tok/s; 1520 tgt tok/s; 1514090455 s elapsed
    Epoch  0,     8/   10; acc:  26.32; ppl: 274.79; 1606 src tok/s; 1545 tgt tok/s; 1514090455 s elapsed
    Epoch  0,     9/   10; acc:  26.83; ppl: 247.32; 1669 src tok/s; 1614 tgt tok/s; 1514090455 s elapsed
    Validation
    Epoch  0,    11/   10; acc:  13.41; ppl: 111.94;   0 src tok/s; 7329 tgt tok/s; 1514090464 s elapsed
    Epoch  1,     0/   10; acc:   6.59; ppl: 147.05; 1849 src tok/s; 1743 tgt tok/s; 1514090464 s elapsed
    Epoch  1,     1/   10; acc:  22.10; ppl: 130.66; 2002 src tok/s; 1957 tgt tok/s; 1514090464 s elapsed
    Epoch  1,     2/   10; acc:  20.16; ppl: 122.49; 1748 src tok/s; 1760 tgt tok/s; 1514090464 s elapsed
    Epoch  1,     3/   10; acc:  23.52; ppl: 117.41; 1690 src tok/s; 1698 tgt tok/s; 1514090464 s elapsed
    Epoch  1,     4/   10; acc:  24.16; ppl: 119.42; 1647 src tok/s; 1662 tgt tok/s; 1514090464 s elapsed
    Epoch  1,     5/   10; acc:  25.44; ppl: 115.31; 1775 src tok/s; 1709 tgt tok/s; 1514090465 s elapsed
    Epoch  1,     6/   10; acc:  24.05; ppl: 115.11; 1780 src tok/s; 1718 tgt tok/s; 1514090465 s elapsed
    Epoch  1,     7/   10; acc:  25.32; ppl: 109.59; 1799 src tok/s; 1765 tgt tok/s; 1514090465 s elapsed
    Epoch  1,     8/   10; acc:  25.14; ppl: 108.16; 1771 src tok/s; 1734 tgt tok/s; 1514090465 s elapsed
    Epoch  1,     9/   10; acc:  25.58; ppl: 107.13; 1817 src tok/s; 1757 tgt tok/s; 1514090465 s elapsed
    Validation
    Epoch  1,    11/   10; acc:  19.58; ppl:  88.09;   0 src tok/s; 7371 tgt tok/s; 1514090474 s elapsed


To use the model, we need to load up the translation functions


```python
import onmt.translate
```


```python
translator = onmt.translate.Translator(beam_size=10, fields=data.fields, model=model)
builder = onmt.translate.TranslationBuilder(data=valid_data, fields=data.fields)

valid_data.src_vocabs
for batch in valid_iter:
    trans_batch = translator.translate_batch(batch=batch, data=valid_data)
    translations = builder.from_batch(trans_batch)
    for trans in translations:
        print(trans.log(0))
    break
```

    PRED SCORE: -4.0690

    SENT 0: ('The', 'competitors', 'have', 'other', 'advantages', ',', 'too', '.')
    PRED 0: .

    PRED SCORE: -4.2736

    SENT 0: ('The', 'company', '&apos;s', 'durability', 'goes', 'back', 'to', 'its', 'first', 'boss', ',', 'a', 'visionary', ',', 'Thomas', 'J.', 'Watson', 'Sr.')
    PRED 0: .

    PRED SCORE: -4.0144

    SENT 0: ('&quot;', 'From', 'what', 'we', 'know', 'today', ',', 'you', 'have', 'to', 'ask', 'how', 'I', 'could', 'be', 'so', 'wrong', '.', '&quot;')
    PRED 0: .

    PRED SCORE: -4.1361

    SENT 0: ('Boeing', 'Co', 'shares', 'rose', '1.5%', 'to', '$', '67.94', '.')
    PRED 0: .

    PRED SCORE: -4.1382

    SENT 0: ('Some', 'did', 'not', 'believe', 'him', ',', 'they', 'said', 'that', 'he', 'got', 'dizzy', 'even', 'in', 'the', 'truck', ',', 'but', 'always', 'wanted', 'to', 'fulfill', 'his', 'dream', ',', 'that', 'of', 'becoming', 'a', 'pilot', '.')
    PRED 0: .

    PRED SCORE: -3.8881

    SENT 0: ('In', 'your', 'opinion', ',', 'the', 'council', 'should', 'ensure', 'that', 'the', 'band', 'immediately', 'above', 'the', 'Ronda', 'de', 'Dalt', 'should', 'provide', 'in', 'its', 'entirety', ',', 'an', 'area', 'of', 'equipment', 'to', 'conduct', 'a', 'smooth', 'transition', 'between', 'the', 'city', 'and', 'the', 'green', '.')
    PRED 0: .

    PRED SCORE: -4.0778

    SENT 0: ('The', 'clerk', 'of', 'the', 'court', ',', 'Jorge', 'Yanez', ',', 'went', 'to', 'the', 'jail', 'of', 'the', 'municipality', 'of', 'San', 'Nicolas', 'of', 'Garza', 'to', 'notify', 'Jonah', 'that', 'he', 'has', 'been', 'legally', 'pardoned', 'and', 'his', 'record', 'will', 'be', 'filed', '.')
    PRED 0: .

    PRED SCORE: -4.2479

    SENT 0: ('&quot;', 'In', 'a', 'research', 'it', 'is', 'reported', 'that', 'there', 'are', 'no', 'parts', 'or', 'components', 'of', 'the', 'ship', 'in', 'another', 'place', ',', 'the', 'impact', 'is', 'presented', 'in', 'a', 'structural', 'way', '.')
    PRED 0: .

    PRED SCORE: -3.8585

    SENT 0: ('On', 'the', 'asphalt', 'covering', ',', 'he', 'added', ',', 'is', 'placed', 'a', 'final', 'layer', 'called', 'rolling', 'covering', ',', 'which', 'is', 'made', '\u200b', '\u200b', 'of', 'a', 'fine', 'stone', 'material', ',', 'meaning', 'sand', 'also', 'dipped', 'into', 'the', 'asphalt', '.')
    PRED 0: .

    PRED SCORE: -4.2298

    SENT 0: ('This', 'is', '200', 'bar', 'on', 'leaving', 'and', '100', 'bar', 'on', 'arrival', '.')
    PRED 0: .



    /usr/local/lib/python3.5/dist-packages/torch/tensor.py:297: UserWarning: other is not broadcastable to self, but they have the same number of elements.  Falling back to deprecated pointwise behavior.
      return self.add_(other)
