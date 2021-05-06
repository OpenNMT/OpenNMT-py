
# Library

The example notebook (available [here](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/examples/Library.ipynb)) should be able to run as a standalone execution, provided `onmt` is in the path (installed via `pip` for instance).

Some parts may not be 100% 'library-friendly' but it's mostly workable.

### Import a few modules and functions that will be necessary


```python
import yaml
import torch
import torch.nn as nn
from argparse import Namespace
from collections import defaultdict, Counter
```


```python
import onmt
from onmt.inputters.inputter import _load_vocab, _build_fields_vocab, get_fields, IterOnDevice
from onmt.inputters.corpus import ParallelCorpus
from onmt.inputters.dynamic_iterator import DynamicDatasetIter
from onmt.translate import GNMTGlobalScorer, Translator, TranslationBuilder
from onmt.utils.misc import set_random_seed
```

### Enable logging


```python
# enable logging
from onmt.utils.logging import init_logger, logger
init_logger()
```




    <RootLogger root (INFO)>



### Set random seed


```python
is_cuda = torch.cuda.is_available()
set_random_seed(1111, is_cuda)
```

### Retrieve data

To make a proper example, we will need some data, as well as some vocabulary(ies).

Let's take the same data as in the [quickstart](https://opennmt.net/OpenNMT-py/quickstart.html):


```python
!wget https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
```

    --2020-09-25 15:28:05--  https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
    Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.217.18.38
    Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.217.18.38|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1662081 (1,6M) [application/x-gzip]
    Saving to: ‘toy-ende.tar.gz.5’
    
    toy-ende.tar.gz.5   100%[===================>]   1,58M  2,33MB/s    in 0,7s    
    
    2020-09-25 15:28:07 (2,33 MB/s) - ‘toy-ende.tar.gz.5’ saved [1662081/1662081]
    



```python
!tar xf toy-ende.tar.gz
```


```python
ls toy-ende
```

    config.yaml  src-test.txt   src-val.txt   tgt-train.txt
    [0m[01;34mrun[0m/         src-train.txt  tgt-test.txt  tgt-val.txt


### Prepare data and vocab

As for any use case of OpenNMT-py 2.0, we can start by creating a simple YAML configuration with our datasets. This is the easiest way to build the proper `opts` `Namespace` that will be used to create the vocabulary(ies).


```python
yaml_config = """
## Where the samples will be written
save_data: toy-ende/run/example
## Where the vocab(s) will be written
src_vocab: toy-ende/run/example.vocab.src
tgt_vocab: toy-ende/run/example.vocab.tgt
# Corpus opts:
data:
    corpus:
        path_src: toy-ende/src-train.txt
        path_tgt: toy-ende/tgt-train.txt
        transforms: []
        weight: 1
    valid:
        path_src: toy-ende/src-val.txt
        path_tgt: toy-ende/tgt-val.txt
        transforms: []
"""
config = yaml.safe_load(yaml_config)
with open("toy-ende/config.yaml", "w") as f:
    f.write(yaml_config)
```


```python
from onmt.utils.parse import ArgumentParser
parser = DynamicArgumentParser(description='build_vocab.py')
```


```python
from onmt.opts import dynamic_prepare_opts
dynamic_prepare_opts(parser, build_vocab_only=True)
```


```python
base_args = (["-config", "toy-ende/config.yaml", "-n_sample", "10000"])
opts, unknown = parser.parse_known_args(base_args)
```


```python
opts
```




    Namespace(config='toy-ende/config.yaml', data="{'corpus': {'path_src': 'toy-ende/src-train.txt', 'path_tgt': 'toy-ende/tgt-train.txt', 'transforms': [], 'weight': 1}, 'valid': {'path_src': 'toy-ende/src-val.txt', 'path_tgt': 'toy-ende/tgt-val.txt', 'transforms': []}}", insert_ratio=0.0, mask_length='subword', mask_ratio=0.0, n_sample=10000, src_onmttok_kwargs="{'mode': 'none'}", tgt_onmttok_kwargs="{'mode': 'none'}", overwrite=False, permute_sent_ratio=0.0, poisson_lambda=0.0, random_ratio=0.0, replace_length=-1, rotate_ratio=0.5, save_config=None, save_data='toy-ende/run/example', seed=-1, share_vocab=False, skip_empty_level='warning', src_seq_length=200, src_subword_model=None, src_subword_type='none', src_vocab=None, subword_alpha=0, subword_nbest=1, switchout_temperature=1.0, tgt_seq_length=200, tgt_subword_model=None, tgt_subword_type='none', tgt_vocab=None, tokendrop_temperature=1.0, tokenmask_temperature=1.0, transforms=[])




```python
from onmt.bin.build_vocab import build_vocab_main
build_vocab_main(opts)
```

    [2020-09-25 15:28:08,068 INFO] Parsed 2 corpora from -data.
    [2020-09-25 15:28:08,069 INFO] Counter vocab from 10000 samples.
    [2020-09-25 15:28:08,070 INFO] Save 10000 transformed example/corpus.
    [2020-09-25 15:28:08,070 INFO] corpus's transforms: TransformPipe()
    [2020-09-25 15:28:08,101 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-25 15:28:08,320 INFO] Just finished the first loop
    [2020-09-25 15:28:08,320 INFO] Counters src:24995
    [2020-09-25 15:28:08,321 INFO] Counters tgt:35816



```python
ls toy-ende/run
```

    example.vocab.src  example.vocab.tgt  [0m[01;34msample[0m/


We just created our source and target vocabularies, respectively `toy-ende/run/example.vocab.src` and `toy-ende/run/example.vocab.tgt`.

### Build fields

We can build the fields from the text files that were just created.


```python
src_vocab_path = "toy-ende/run/example.vocab.src"
tgt_vocab_path = "toy-ende/run/example.vocab.tgt"
```


```python
# initialize the frequency counter
counters = defaultdict(Counter)
# load source vocab
_src_vocab, _src_vocab_size = _load_vocab(
    src_vocab_path,
    'src',
    counters)
# load target vocab
_tgt_vocab, _tgt_vocab_size = _load_vocab(
    tgt_vocab_path,
    'tgt',
    counters)
```

    [2020-09-25 15:28:08,495 INFO] Loading src vocabulary from toy-ende/run/example.vocab.src
    [2020-09-25 15:28:08,554 INFO] Loaded src vocab has 24995 tokens.
    [2020-09-25 15:28:08,562 INFO] Loading tgt vocabulary from toy-ende/run/example.vocab.tgt
    [2020-09-25 15:28:08,617 INFO] Loaded tgt vocab has 35816 tokens.



```python
# initialize fields
src_nfeats, tgt_nfeats = 0, 0 # do not support word features for now
fields = get_fields(
    'text', src_nfeats, tgt_nfeats)
```


```python
fields
```




    {'src': <onmt.inputters.text_dataset.TextMultiField at 0x7fca93802c50>,
     'tgt': <onmt.inputters.text_dataset.TextMultiField at 0x7fca93802f60>,
     'indices': <torchtext.data.field.Field at 0x7fca93802940>}




```python
# build fields vocab
share_vocab = False
vocab_size_multiple = 1
src_vocab_size = 30000
tgt_vocab_size = 30000
src_words_min_frequency = 1
tgt_words_min_frequency = 1
vocab_fields = _build_fields_vocab(
    fields, counters, 'text', share_vocab,
    vocab_size_multiple,
    src_vocab_size, src_words_min_frequency,
    tgt_vocab_size, tgt_words_min_frequency)
```

    [2020-09-25 15:28:08,699 INFO]  * tgt vocab size: 30004.
    [2020-09-25 15:28:08,749 INFO]  * src vocab size: 24997.


An alternative way of creating these fields is to run `onmt_train` without actually training, to just output the necessary files.

### Prepare for training: model and optimizer creation

Let's get a few fields/vocab related variables to simplify the model creation a bit:


```python
src_text_field = vocab_fields["src"].base_field
src_vocab = src_text_field.vocab
src_padding = src_vocab.stoi[src_text_field.pad_token]

tgt_text_field = vocab_fields['tgt'].base_field
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
    nn.LogSoftmax(dim=-1)).to(device)

loss = onmt.utils.loss.NMTLossCompute(
    criterion=nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"),
    generator=model.generator)
```

Now we set up the optimizer. This could be a core torch optim class, or our wrapper which handles learning rate updates and gradient normalization automatically.


```python
lr = 1
torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optim = onmt.utils.optimizers.Optimizer(
    torch_optimizer, learning_rate=lr, max_grad_norm=2)
```

### Create the training and validation data iterators

Now we need to create the dynamic dataset iterator.

This is not very 'library-friendly' for now because of the way the `DynamicDatasetIter` constructor is defined. It may evolve in the future.


```python
src_train = "toy-ende/src-train.txt"
tgt_train = "toy-ende/tgt-train.txt"
src_val = "toy-ende/src-val.txt"
tgt_val = "toy-ende/tgt-val.txt"

# build the ParallelCorpus
corpus = ParallelCorpus("corpus", src_train, tgt_train)
valid = ParallelCorpus("valid", src_val, tgt_val)
```


```python
# build the training iterator
train_iter = DynamicDatasetIter(
    corpora={"corpus": corpus},
    corpora_info={"corpus": {"weight": 1}},
    transforms={},
    fields=vocab_fields,
    is_train=True,
    batch_type="tokens",
    batch_size=4096,
    batch_size_multiple=1,
    data_type="text")
```


```python
# make sure the iteration happens on GPU 0 (-1 for CPU, N for GPU N)
train_iter = iter(IterOnDevice(train_iter, 0))
```


```python
# build the validation iterator
valid_iter = DynamicDatasetIter(
    corpora={"valid": valid},
    corpora_info={"valid": {"weight": 1}},
    transforms={},
    fields=vocab_fields,
    is_train=False,
    batch_type="sents",
    batch_size=8,
    batch_size_multiple=1,
    data_type="text")
```


```python
valid_iter = IterOnDevice(valid_iter, 0)
```

### Training

Finally we train.


```python
report_manager = onmt.utils.ReportMgr(
    report_every=50, start_time=None, tensorboard_writer=None)

trainer = onmt.Trainer(model=model,
                       train_loss=loss,
                       valid_loss=loss,
                       optim=optim,
                       report_manager=report_manager,
                       dropout=[0.1])

trainer.train(train_iter=train_iter,
              train_steps=1000,
              valid_iter=valid_iter,
              valid_steps=500)
```

    [2020-09-25 15:28:15,184 INFO] Start training loop and validate every 500 steps...
    [2020-09-25 15:28:15,185 INFO] corpus's transforms: TransformPipe()
    [2020-09-25 15:28:15,187 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-25 15:28:21,140 INFO] Step 50/ 1000; acc:   7.52; ppl: 8832.29; xent: 9.09; lr: 1.00000; 18916/18871 tok/s;      6 sec
    [2020-09-25 15:28:24,869 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-25 15:28:27,121 INFO] Step 100/ 1000; acc:   9.34; ppl: 1840.06; xent: 7.52; lr: 1.00000; 18911/18785 tok/s;     12 sec
    [2020-09-25 15:28:33,048 INFO] Step 150/ 1000; acc:  10.35; ppl: 1419.18; xent: 7.26; lr: 1.00000; 19062/19017 tok/s;     18 sec
    [2020-09-25 15:28:37,019 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-25 15:28:39,022 INFO] Step 200/ 1000; acc:  11.14; ppl: 1127.44; xent: 7.03; lr: 1.00000; 19084/18911 tok/s;     24 sec
    [2020-09-25 15:28:45,073 INFO] Step 250/ 1000; acc:  12.46; ppl: 912.13; xent: 6.82; lr: 1.00000; 18575/18570 tok/s;     30 sec
    [2020-09-25 15:28:49,301 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-25 15:28:51,151 INFO] Step 300/ 1000; acc:  13.04; ppl: 779.50; xent: 6.66; lr: 1.00000; 18394/18307 tok/s;     36 sec
    [2020-09-25 15:28:57,316 INFO] Step 350/ 1000; acc:  14.04; ppl: 685.48; xent: 6.53; lr: 1.00000; 18339/18173 tok/s;     42 sec
    [2020-09-25 15:29:02,117 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-25 15:29:03,576 INFO] Step 400/ 1000; acc:  14.99; ppl: 590.20; xent: 6.38; lr: 1.00000; 18090/18029 tok/s;     48 sec
    [2020-09-25 15:29:09,546 INFO] Step 450/ 1000; acc:  16.00; ppl: 524.51; xent: 6.26; lr: 1.00000; 18726/18536 tok/s;     54 sec
    [2020-09-25 15:29:14,585 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-25 15:29:15,596 INFO] Step 500/ 1000; acc:  16.78; ppl: 453.38; xent: 6.12; lr: 1.00000; 17877/17980 tok/s;     60 sec
    [2020-09-25 15:29:15,597 INFO] valid's transforms: TransformPipe()
    [2020-09-25 15:29:15,599 INFO] Loading ParallelCorpus(toy-ende/src-val.txt, toy-ende/tgt-val.txt, align=None)...
    [2020-09-25 15:29:24,528 INFO] Validation perplexity: 295.1
    [2020-09-25 15:29:24,529 INFO] Validation accuracy: 17.6533
    [2020-09-25 15:29:30,592 INFO] Step 550/ 1000; acc:  17.47; ppl: 421.26; xent: 6.04; lr: 1.00000; 7726/7610 tok/s;     75 sec
    [2020-09-25 15:29:36,055 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-25 15:29:36,695 INFO] Step 600/ 1000; acc:  18.95; ppl: 354.44; xent: 5.87; lr: 1.00000; 17470/17598 tok/s;     82 sec
    [2020-09-25 15:29:42,794 INFO] Step 650/ 1000; acc:  19.60; ppl: 328.47; xent: 5.79; lr: 1.00000; 18994/18793 tok/s;     88 sec
    [2020-09-25 15:29:48,635 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-25 15:29:48,924 INFO] Step 700/ 1000; acc:  20.57; ppl: 285.48; xent: 5.65; lr: 1.00000; 17856/17788 tok/s;     94 sec
    [2020-09-25 15:29:54,898 INFO] Step 750/ 1000; acc:  21.97; ppl: 249.06; xent: 5.52; lr: 1.00000; 19030/18924 tok/s;    100 sec
    [2020-09-25 15:30:01,233 INFO] Step 800/ 1000; acc:  22.66; ppl: 228.54; xent: 5.43; lr: 1.00000; 17571/17471 tok/s;    106 sec
    [2020-09-25 15:30:01,357 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-25 15:30:07,345 INFO] Step 850/ 1000; acc:  24.32; ppl: 193.65; xent: 5.27; lr: 1.00000; 18344/18313 tok/s;    112 sec
    [2020-09-25 15:30:11,363 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-25 15:30:13,487 INFO] Step 900/ 1000; acc:  24.93; ppl: 177.65; xent: 5.18; lr: 1.00000; 18293/18105 tok/s;    118 sec
    [2020-09-25 15:30:19,670 INFO] Step 950/ 1000; acc:  26.33; ppl: 157.10; xent: 5.06; lr: 1.00000; 17791/17746 tok/s;    124 sec
    [2020-09-25 15:30:24,072 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-25 15:30:25,820 INFO] Step 1000/ 1000; acc:  27.47; ppl: 137.64; xent: 4.92; lr: 1.00000; 17942/17962 tok/s;    131 sec
    [2020-09-25 15:30:25,822 INFO] Loading ParallelCorpus(toy-ende/src-val.txt, toy-ende/tgt-val.txt, align=None)...
    [2020-09-25 15:30:34,665 INFO] Validation perplexity: 241.801
    [2020-09-25 15:30:34,666 INFO] Validation accuracy: 20.2837





    <onmt.utils.statistics.Statistics at 0x7fca934e8e80>



### Translate

For translation, we can build a "traditional" (as opposed to dynamic) dataset for now.


```python
src_data = {"reader": onmt.inputters.str2reader["text"](), "data": src_val}
tgt_data = {"reader": onmt.inputters.str2reader["text"](), "data": tgt_val}
_readers, _data = onmt.inputters.Dataset.config(
    [('src', src_data), ('tgt', tgt_data)])
```


```python
dataset = onmt.inputters.Dataset(
    vocab_fields, readers=_readers, data=_data,
    sort_key=onmt.inputters.str2sortkey["text"])
```


```python
data_iter = onmt.inputters.OrderedIterator(
            dataset=dataset,
            device="cuda",
            batch_size=10,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )
```


```python
src_reader = onmt.inputters.str2reader["text"]
tgt_reader = onmt.inputters.str2reader["text"]
scorer = GNMTGlobalScorer(alpha=0.7, 
                          beta=0., 
                          length_penalty="avg", 
                          coverage_penalty="none")
gpu = 0 if torch.cuda.is_available() else -1
translator = Translator(model=model, 
                        fields=vocab_fields, 
                        src_reader=src_reader, 
                        tgt_reader=tgt_reader, 
                        global_scorer=scorer,
                        gpu=gpu)
builder = onmt.translate.TranslationBuilder(data=dataset, 
                                            fields=vocab_fields)
```

**Note**: translations will be very poor, because of the very low quantity of data, the absence of proper tokenization, and the brevity of the training.


```python
for batch in data_iter:
    trans_batch = translator.translate_batch(
        batch=batch, src_vocabs=[src_vocab],
        attn_debug=False)
    translations = builder.from_batch(trans_batch)
    for trans in translations:
        print(trans.log(0))
    break
```

    
    SENT 0: ['Parliament', 'Does', 'Not', 'Support', 'Amendment', 'Freeing', 'Tymoshenko']
    PRED 0: Parlament das Parlament über die Europäische Parlament , die sich in der Lage in der Lage ist , die es in der Lage sind .
    PRED SCORE: -1.5935
    
    
    SENT 0: ['Today', ',', 'the', 'Ukraine', 'parliament', 'dismissed', ',', 'within', 'the', 'Code', 'of', 'Criminal', 'Procedure', 'amendment', ',', 'the', 'motion', 'to', 'revoke', 'an', 'article', 'based', 'on', 'which', 'the', 'opposition', 'leader', ',', 'Yulia', 'Tymoshenko', ',', 'was', 'sentenced', '.']
    PRED 0: In der Nähe des Hotels , die in der Lage , die sich in der Lage ist , in der Lage , die in der Lage , die in der Lage ist .
    PRED SCORE: -1.7173
    
    
    SENT 0: ['The', 'amendment', 'that', 'would', 'lead', 'to', 'freeing', 'the', 'imprisoned', 'former', 'Prime', 'Minister', 'was', 'revoked', 'during', 'second', 'reading', 'of', 'the', 'proposal', 'for', 'mitigation', 'of', 'sentences', 'for', 'economic', 'offences', '.']
    PRED 0: Die Tatsache , die sich in der Lage in der Lage ist , die für eine Antwort der Entwicklung für die Entwicklung von Präsident .
    PRED SCORE: -1.6834
    
    
    SENT 0: ['In', 'October', ',', 'Tymoshenko', 'was', 'sentenced', 'to', 'seven', 'years', 'in', 'prison', 'for', 'entering', 'into', 'what', 'was', 'reported', 'to', 'be', 'a', 'disadvantageous', 'gas', 'deal', 'with', 'Russia', '.']
    PRED 0: In der Nähe wurde die Menschen in der Lage ist , die sich in der Lage <unk> .
    PRED SCORE: -1.5765
    
    
    SENT 0: ['The', 'verdict', 'is', 'not', 'yet', 'final;', 'the', 'court', 'will', 'hear', 'Tymoshenko', '&apos;s', 'appeal', 'in', 'December', '.']
    PRED 0: Es ist nicht der Fall , die in der Lage in der Lage sind .
    PRED SCORE: -1.3287
    
    
    SENT 0: ['Tymoshenko', 'claims', 'the', 'verdict', 'is', 'a', 'political', 'revenge', 'of', 'the', 'regime;', 'in', 'the', 'West', ',', 'the', 'trial', 'has', 'also', 'evoked', 'suspicion', 'of', 'being', 'biased', '.']
    PRED 0: Um in der Lage ist auch eine Lösung Rolle .
    PRED SCORE: -1.3975
    
    
    SENT 0: ['The', 'proposal', 'to', 'remove', 'Article', '365', 'from', 'the', 'Code', 'of', 'Criminal', 'Procedure', ',', 'upon', 'which', 'the', 'former', 'Prime', 'Minister', 'was', 'sentenced', ',', 'was', 'supported', 'by', '147', 'members', 'of', 'parliament', '.']
    PRED 0: Der Vorschlag , die in der Lage , die in der Lage , die in der Lage ist , war er von der Fall <unk> wurde .
    PRED SCORE: -1.6062
    
    
    SENT 0: ['Its', 'ratification', 'would', 'require', '226', 'votes', '.']
    PRED 0: Es wäre noch einmal noch einmal <unk> .
    PRED SCORE: -1.8001
    
    
    SENT 0: ['Libya', '&apos;s', 'Victory']
    PRED 0: In der Nähe des Hotels befindet sich in der Nähe des Hotels in der Lage .
    PRED SCORE: -1.7097
    
    
    SENT 0: ['The', 'story', 'of', 'Libya', '&apos;s', 'liberation', ',', 'or', 'rebellion', ',', 'already', 'has', 'its', 'defeated', '.']
    PRED 0: In der Nähe des Hotels in der Lage ist in der Lage .
    PRED SCORE: -1.7885
