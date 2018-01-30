import argparse
import copy
import unittest
import glob
import os
from collections import Counter

import torchtext

import onmt
import onmt.io
import opts
import preprocess


parser = argparse.ArgumentParser(description='preprocess.py')
opts.preprocess_opts(parser)


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
        fields = onmt.io.get_fields("text", 0, 0)

        train_data_files = preprocess.build_save_dataset('train', fields, opt)

        preprocess.build_save_vocab(train_data_files, fields, opt)

        preprocess.build_save_dataset('valid', fields, opt)

        # Remove the generated *pt files.
        for pt in glob.glob(SAVE_DATA_PREFIX + '*.pt'):
            os.remove(pt)

    def test_merge_vocab(self):
        va = torchtext.vocab.Vocab(Counter('abbccc'))
        vb = torchtext.vocab.Vocab(Counter('eeabbcccf'))

        merged = onmt.io.merge_vocabs([va, vb], 2)

        self.assertEqual(Counter({'c': 6, 'b': 4, 'a': 2, 'e': 2, 'f': 1}),
                         merged.freqs)
        # 4 specicials + 2 words (since we pass 2 to merge_vocabs)
        self.assertEqual(6, len(merged.itos))
        self.assertTrue('b' in merged.itos)


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
                  [('src_seq_length', 1)],
                  [('src_seq_length', 5000)],
                  [('src_seq_length_trunc', 1)],
                  [('src_seq_length_trunc', 5000)],
                  [('tgt_seq_length', 1)],
                  [('tgt_seq_length', 5000)],
                  [('tgt_seq_length_trunc', 1)],
                  [('tgt_seq_length_trunc', 5000)],
                  [('shuffle', 0)],
                  [('lower', True)],
                  [('dynamic_dict', True)],
                  [('share_vocab', True)],
                  [('dynamic_dict', True),
                   ('share_vocab', True)],
                  [('dynamic_dict', True),
                   ('max_shard_size', 500000)],
                  ]

for p in test_databuild:
    _add_test(p, 'dataset_build')

# Test image preprocessing
for p in copy.deepcopy(test_databuild):
    p.append(('data_type', 'img'))
    p.append(('src_dir', '/tmp/im2text/images'))
    p.append(('train_src', '/tmp/im2text/src-train-head.txt'))
    p.append(('train_tgt', '/tmp/im2text/tgt-train-head.txt'))
    p.append(('valid_src', '/tmp/im2text/src-val-head.txt'))
    p.append(('valid_tgt', '/tmp/im2text/tgt-val-head.txt'))
    _add_test(p, 'dataset_build')

# Test audio preprocessing
for p in copy.deepcopy(test_databuild):
    p.append(('data_type', 'audio'))
    p.append(('src_dir', '/tmp/speech/an4_dataset'))
    p.append(('train_src', '/tmp/speech/src-train-head.txt'))
    p.append(('train_tgt', '/tmp/speech/tgt-train-head.txt'))
    p.append(('valid_src', '/tmp/speech/src-val-head.txt'))
    p.append(('valid_tgt', '/tmp/speech/tgt-val-head.txt'))
    p.append(('sample_rate', 16000))
    p.append(('window_size', 0.04))
    p.append(('window_stride', 0.02))
    p.append(('window', 'hamming'))
    _add_test(p, 'dataset_build')
