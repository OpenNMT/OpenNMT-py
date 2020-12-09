"""
Language Translation with TorchText
===================================

This tutorial shows how to use ``torchtext`` to preprocess
data from a well-known dataset containing sentences in both English and German and use it to
train a sequence-to-sequence model with attention that can translate German sentences
into English.

It is based off of
`this tutorial <https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb>`__
from PyTorch community member `Ben Trevett <https://github.com/bentrevett>`__
with Ben's permission. We update the tutorials by removing some legacy code.

By the end of this tutorial, you will be able to preprocess sentences into tensors for NLP modeling and use `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader>`__ for training and validing the model.
"""

######################################################################
# Data Processing
# ----------------
# ``torchtext`` has utilities for creating datasets that can be easily
# iterated through for the purposes of creating a language translation
# model. In this example, we show how to tokenize a raw text sentence, build vocabulary, and numericalize tokens into tensor.
#
# Note: the tokenization in this tutorial requires `Spacy <https://spacy.io>`__
# We use Spacy because it provides strong support for tokenization in languages
# other than English. ``torchtext`` provides a ``basic_english`` tokenizer
# and supports other tokenizers for English (e.g.
# `Moses <https://bitbucket.org/luismsgomes/mosestokenizer/src/default/>`__)
# but for language translation - where multiple languages are required -
# Spacy is your best bet.
#
# To run this tutorial, first install ``spacy`` using ``pip`` or ``conda``.
# Next, download the raw data for the English and German Spacy tokenizers:
#
# ::
#
#    hl -> C
#    ll -> wat

import torchtext
import torch
from torch.nn import Transformer
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
import io

base_dir = '/mnt/c/Codenator/try_2/datasets/'
train_files_names = ('train0.corpus.ll', 'train0.corpus.hl')
val_files_names = ('validate0.corpus.ll', 'validate0.corpus.hl')
test_files_names = ('test0.corpus.ll', 'test0.corpus.hl')

train_filepaths = [base_dir + file_name for file_name in train_files_names]
val_filepaths = [base_dir + file_name for file_name in val_files_names]
test_filepaths = [base_dir + file_name for file_name in test_files_names]

ll_tokenizer = get_tokenizer(None)
hl_tokenizer = get_tokenizer(None)

def build_vocab(filepath, tokenizer):
  counter = Counter()
  with io.open(filepath, encoding="utf8") as f:
    for string_ in f:
      counter.update(tokenizer(string_))
  return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

ll_vocab = build_vocab(train_filepaths[0], ll_tokenizer)
hl_vocab = build_vocab(train_filepaths[1], hl_tokenizer)

def data_process(filepaths):
  raw_ll_iter = iter(io.open(filepaths[0], encoding="utf8"))
  raw_hl_iter = iter(io.open(filepaths[1], encoding="utf8"))
  data = []
  counter = 0
  for (raw_ll, raw_hl) in zip(raw_ll_iter, raw_hl_iter):
    ll_tensor_ = torch.tensor([ll_vocab[token] for token in ll_tokenizer(raw_ll)],
                              dtype=torch.long)
    hl_tensor_ = torch.tensor([hl_vocab[token] for token in hl_tokenizer(raw_hl)],
                              dtype=torch.long)
    data.append((ll_tensor_, hl_tensor_))
    counter += 1
    if counter > 2:
        break
  return data

train_data = data_process(train_filepaths)
val_data = data_process(val_filepaths)
test_data = data_process(test_filepaths)

######################################################################
# ``DataLoader``
# ----------------
# The last ``torch`` specific feature we'll use is the ``DataLoader``,
# which is easy to use since it takes the data as its
# first argument. Specifically, as the docs say:
# ``DataLoader`` combines a dataset and a sampler, and provides an iterable over the given dataset. The ``DataLoader`` supports both map-style and iterable-style datasets with single- or multi-process loading, customizing loading order and optional automatic batching (collation) and memory pinning.
#
# Please pay attention to ``collate_fn`` (optional) that merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.
#

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 8
PAD_IDX = ll_vocab['<pad>']
BOS_IDX = ll_vocab['<bos>']
EOS_IDX = ll_vocab['<eos>']

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def generate_batch(data_batch):
  ll_batch, hl_batch = [], []
  for (ll_item, hl_item) in data_batch:
    ll_batch.append(torch.cat([torch.tensor([BOS_IDX]), ll_item, torch.tensor([EOS_IDX])], dim=0))
    hl_batch.append(torch.cat([torch.tensor([BOS_IDX]), hl_item, torch.tensor([EOS_IDX])], dim=0))
  ll_batch = pad_sequence(ll_batch, padding_value=PAD_IDX)
  hl_batch = pad_sequence(hl_batch, padding_value=PAD_IDX)
  return ll_batch, hl_batch

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)


######################################################################
# Defining our ``nn.Module`` and ``Optimizer``
# ----------------
# That's mostly it from a ``torchtext`` perspecive: with the dataset built
# and the iterator defined, the rest of this tutorial simply defines our
# model as an ``nn.Module``, along with an ``Optimizer``, and then trains it.
#
# Our model specifically, follows the architecture described
# `here <https://arxiv.org/abs/1409.0473>`__ (you can find a
# significantly more commented version
# `here <https://github.com/SethHWeidman/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb>`__).
#
# Note: this model is just an example model that can be used for language
# translation; we choose it because it is a standard model for the task,
# not because it is the recommended model to use for translation. As you're
# likely aware, state-of-the-art models are currently based on Transformers;
# you can see PyTorch's capabilities for implementing Transformer layers
# `here <https://pytorch.org/docs/stable/nn.html#transformer-layers>`__; and
# in particular, the "attention" used in the model below is different from
# the multi-headed self-attention present in a transformer model.


import random
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

m_dim = 32
nhead = 2
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward=64

model = Transformer(d_model=m_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward).to(device)
model2 = Transformer(d_model=m_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward).to(device)

def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)
model2.apply(init_weights)

optimizer = optim.Adam(model.parameters())


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

######################################################################
# Note: when scoring the performance of a language translation model in
# particular, we have to tell the ``nn.CrossEntropyLoss`` function to
# ignore the indices where the target is simply padding.

PAD_IDX = hl_vocab.stoi['<pad>']

criterion = nn.L1Loss()

######################################################################
# Finally, we can train and evaluate this model:

import math
import time

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def train(model: nn.Module,
          model2: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          pos_encoding: PositionalEncoding,
          clip: float,
          ll_embedder: nn.Module,
          hl_embedder: nn.Module,
          linear_decoder: nn.Module):

    model.train()

    epoch_loss = 0

    for _, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        or_trg = trg

        optimizer.zero_grad()
        src = ll_embedder(src) * math.sqrt(m_dim)
        src = pos_encoding(src)

        trg = hl_embedder(trg) * math.sqrt(m_dim)
        trg = pos_encoding(trg)

        output = model(src, trg)
        output = model2(output, trg)
        # print(output.shape)
        # output = linear_decoder(output)
        # print(output.shape)
        # print(trg.shape)
        # output = output[1:].view(-1, output.shape[-1])
        # trg = trg[1:].view(-1)

        print(output.shape)
        print(trg.shape)
        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module,
             is_translation: int):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 1
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, model2, train_iter, optimizer, criterion, PositionalEncoding(m_dim), CLIP,
                       nn.Embedding(len(ll_vocab), m_dim), nn.Embedding(len(hl_vocab), m_dim), nn.Linear(m_dim, len(hl_vocab)))
    valid_loss = evaluate(model, valid_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iter, criterion)
max_len = 10


print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
