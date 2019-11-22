#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import configargparse
import copy
import unittest
import glob
import os
import codecs

import onmt
import onmt.inputters
import onmt.opts
import onmt.bin.preprocess as preprocess


parser = configargparse.ArgumentParser(description='preprocess.py')
onmt.opts.preprocess_opts(parser)

SAVE_DATA_PREFIX = 'data/test_preprocess'

default_opts = [
    '-data_type', 'text',
    '-train_src', 'data/src-train.txt',
    '-train_tgt', 'data/tgt-train.txt',
    '-valid_src', 'data/src-val.txt',
    '-valid_tgt', 'data/tgt-val.txt',
    '-save_data', SAVE_DATA_PREFIX
]

opt = parser.parse_known_args(default_opts)[0]


class TestData(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestData, self).__init__(*args, **kwargs)
        self.opt = opt

    def dataset_build(self, opt):
        fields = onmt.inputters.get_fields("text", 0, 0)

        if hasattr(opt, 'src_vocab') and len(opt.src_vocab) > 0:
            with codecs.open(opt.src_vocab, 'w', 'utf-8') as f:
                f.write('a\nb\nc\nd\ne\nf\n')
        if hasattr(opt, 'tgt_vocab') and len(opt.tgt_vocab) > 0:
            with codecs.open(opt.tgt_vocab, 'w', 'utf-8') as f:
                f.write('a\nb\nc\nd\ne\nf\n')

        src_reader = onmt.inputters.str2reader[opt.data_type].from_opt(opt)
        tgt_reader = onmt.inputters.str2reader["text"].from_opt(opt)
        align_reader = onmt.inputters.str2reader["text"].from_opt(opt)
        preprocess.build_save_dataset(
            'train', fields, src_reader, tgt_reader, align_reader, opt)

        preprocess.build_save_dataset(
            'valid', fields, src_reader, tgt_reader, align_reader, opt)

        # Remove the generated *pt files.
        for pt in glob.glob(SAVE_DATA_PREFIX + '*.pt'):
            os.remove(pt)
        if hasattr(opt, 'src_vocab') and os.path.exists(opt.src_vocab):
            os.remove(opt.src_vocab)
        if hasattr(opt, 'tgt_vocab') and os.path.exists(opt.tgt_vocab):
            os.remove(opt.tgt_vocab)


def _add_test(param_setting, methodname):
    """
    Adds a Test to TestData according to settings

    Args:
        param_setting: list of tuples of (param, setting)
        methodname: name of the method that gets called
    """

    def test_method(self):
        if param_setting:
            opt = copy.deepcopy(self.opt)
            for param, setting in param_setting:
                setattr(opt, param, setting)
        else:
            opt = self.opt
        getattr(self, methodname)(opt)
    if param_setting:
        name = 'test_' + methodname + "_" + "_".join(
            str(param_setting).split())
    else:
        name = 'test_' + methodname + '_standard'
    setattr(TestData, name, test_method)
    test_method.__name__ = name


test_databuild = [[],
                  [('src_vocab_size', 1),
                   ('tgt_vocab_size', 1)],
                  [('src_vocab_size', 10000),
                   ('tgt_vocab_size', 10000)],
                  [('src_seq_len', 1)],
                  [('src_seq_len', 5000)],
                  [('src_seq_length_trunc', 1)],
                  [('src_seq_length_trunc', 5000)],
                  [('tgt_seq_len', 1)],
                  [('tgt_seq_len', 5000)],
                  [('tgt_seq_length_trunc', 1)],
                  [('tgt_seq_length_trunc', 5000)],
                  [('shuffle', 0)],
                  [('lower', True)],
                  [('dynamic_dict', True)],
                  [('share_vocab', True)],
                  [('dynamic_dict', True),
                   ('share_vocab', True)],
                  [('dynamic_dict', True),
                   ('shard_size', 500000)],
                  [('src_vocab', '/tmp/src_vocab.txt'),
                   ('tgt_vocab', '/tmp/tgt_vocab.txt')],
                  ]

for p in test_databuild:
    _add_test(p, 'dataset_build')

# Test image preprocessing
test_databuild = [[],
                  [('tgt_vocab_size', 1)],
                  [('tgt_vocab_size', 10000)],
                  [('tgt_seq_len', 1)],
                  [('tgt_seq_len', 5000)],
                  [('tgt_seq_length_trunc', 1)],
                  [('tgt_seq_length_trunc', 5000)],
                  [('shuffle', 0)],
                  [('lower', True)],
                  [('shard_size', 5)],
                  [('shard_size', 50)],
                  [('tgt_vocab', '/tmp/tgt_vocab.txt')],
                  ]
test_databuild_common = [('data_type', 'img'),
                         ('src_dir', '/tmp/im2text/images'),
                         ('train_src', ['/tmp/im2text/src-train-head.txt']),
                         ('train_tgt', ['/tmp/im2text/tgt-train-head.txt']),
                         ('valid_src', '/tmp/im2text/src-val-head.txt'),
                         ('valid_tgt', '/tmp/im2text/tgt-val-head.txt'),
                         ]
for p in test_databuild:
    _add_test(p + test_databuild_common, 'dataset_build')

# Test audio preprocessing
test_databuild = [[],
                  [('tgt_vocab_size', 1)],
                  [('tgt_vocab_size', 10000)],
                  [('src_seq_len', 1)],
                  [('src_seq_len', 5000)],
                  [('src_seq_length_trunc', 3200)],
                  [('src_seq_length_trunc', 5000)],
                  [('tgt_seq_len', 1)],
                  [('tgt_seq_len', 5000)],
                  [('tgt_seq_length_trunc', 1)],
                  [('tgt_seq_length_trunc', 5000)],
                  [('shuffle', 0)],
                  [('lower', True)],
                  [('shard_size', 5)],
                  [('shard_size', 50)],
                  [('tgt_vocab', '/tmp/tgt_vocab.txt')],
                  ]
test_databuild_common = [('data_type', 'audio'),
                         ('src_dir', '/tmp/speech/an4_dataset'),
                         ('train_src', ['/tmp/speech/src-train-head.txt']),
                         ('train_tgt', ['/tmp/speech/tgt-train-head.txt']),
                         ('valid_src', '/tmp/speech/src-val-head.txt'),
                         ('valid_tgt', '/tmp/speech/tgt-val-head.txt'),
                         ('sample_rate', 16000),
                         ('window_size', 0.04),
                         ('window_stride', 0.02),
                         ('window', 'hamming'),
                         ]
for p in test_databuild:
    _add_test(p + test_databuild_common, 'dataset_build')
