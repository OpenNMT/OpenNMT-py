#!/usr/bin/env python
import torch
import imp
import argparse
import pyonmttok
from onmt.constants import DefaultTokens
from onmt.inputters.inputter import vocabs_to_dict


def _feature_tokenize():
    return 0


class RawField(object):
    def __init__(self):
        pass


class TextMultiField(RawField):
    def __init__(self):
        pass


class Field(RawField):
    def __init__(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v2model', type=str, required=True)
    parser.add_argument('-v3model', type=str, required=True)

    opt = parser.parse_args()
    print(opt)
    module1 = imp.load_source("torchtext.data.field", "tools/convertv2_v3.py")
    module2 = imp.load_source("onmt.inputters.text_dataset",
                              "tools/convertv2_v3.py")
    # module3 = imp.load_source("Vocab", "tools/convertv2_v3.py")
    checkpoint = torch.load(opt.v2model)
    vocabs = {}
    multifield = checkpoint['vocab']['src']
    multifields = multifield.fields
    _, fields = multifields[0]
    voc = dict(sorted(fields.vocab.__dict__['freqs'].items(),
                      key=lambda x: (-x[1], x[0]))).keys()
    src_vocab = pyonmttok.build_vocab_from_tokens(
        voc,
        maximum_size=0,
        minimum_frequency=1,
        special_tokens=[DefaultTokens.UNK,
                        DefaultTokens.PAD,
                        DefaultTokens.BOS,
                        DefaultTokens.EOS])
    src_vocab.default_id = src_vocab[DefaultTokens.UNK]
    vocabs['src'] = src_vocab
    print("Source vocab size is:", len(src_vocab))
    multifield = checkpoint['vocab']['tgt']
    multifields = multifield.fields
    _, fields = multifields[0]
    voc = dict(sorted(fields.vocab.__dict__['freqs'].items(),
                      key=lambda x: (-x[1], x[0]))).keys()
    tgt_vocab = pyonmttok.build_vocab_from_tokens(
        voc,
        maximum_size=0,
        minimum_frequency=1,
        special_tokens=[DefaultTokens.UNK,
                        DefaultTokens.PAD,
                        DefaultTokens.BOS,
                        DefaultTokens.EOS])
    tgt_vocab.default_id = src_vocab[DefaultTokens.UNK]
    vocabs['tgt'] = tgt_vocab
    print("Target vocab size is:", len(tgt_vocab))
    if hasattr(checkpoint['opt'], 'data_task'):
        print("Model is type:", checkpoint['opt'].data_task)
        vocabs['data_task'] = checkpoint['opt'].data_task
    else:
        vocabs['data_task'] = "seq2seq"
    checkpoint['vocab'] = vocabs_to_dict(vocabs)
    torch.save(checkpoint, opt.v3model)
