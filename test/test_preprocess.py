import argparse
import copy
import unittest
import onmt
import opts
import torchtext

from collections import Counter


parser = argparse.ArgumentParser(description='preprocess.py')
opts.preprocess_opts(parser)

opt = parser.parse_known_args()[0]

opt.data_type = 'text'
opt.train_src = 'data/src-train.txt'
opt.train_tgt = 'data/tgt-train.txt'
opt.valid_src = 'data/src-val.txt'
opt.valid_tgt = 'data/tgt-val.txt'


class TestData(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestData, self).__init__(*args, **kwargs)
        self.opt = opt

    def dataset_build(self, opt):
        fields = onmt.IO.get_fields("text", 0, 0)

        train = onmt.IO.ONMTDataset(
            "text", opt.train_src, opt.train_tgt, fields,
            opt.src_seq_length, opt.tgt_seq_length,
            src_seq_length_trunc=opt.src_seq_length_trunc,
            tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
            dynamic_dict=opt.dynamic_dict)

        onmt.IO.build_vocab(train, opt)

        onmt.IO.ONMTDataset(
            "text", opt.valid_src, opt.valid_tgt, fields,
            opt.src_seq_length, opt.tgt_seq_length,
            src_seq_length_trunc=opt.src_seq_length_trunc,
            tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
            dynamic_dict=opt.dynamic_dict)

    def test_merge_vocab(self):
        va = torchtext.vocab.Vocab(Counter('abbccc'))
        vb = torchtext.vocab.Vocab(Counter('eeabbcccf'))

        merged = onmt.IO.merge_vocabs([va, vb], 2)

        self.assertEqual(Counter({'c': 6, 'b': 4, 'a': 2, 'e': 2, 'f': 1}),
                         merged.freqs)
        self.assertEqual(6, len(merged.itos))
        self.assertTrue('b' in merged.itos)


def _add_test(paramSetting, methodname):
    """
    Adds a Test to TestData according to settings

    Args:
        paramSetting: list of tuples of (param, setting)
        methodname: name of the method that gets called
    """

    def test_method(self):
        if paramSetting:
            opt = copy.deepcopy(self.opt)
            for param, setting in paramSetting:
                setattr(opt, param, setting)
        else:
            opt = self.opt
        getattr(self, methodname)(opt)
    if paramSetting:
        name = 'test_' + methodname + "_" + "_".join(str(paramSetting).split())
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
                  ]

for p in test_databuild:
    _add_test(p, 'dataset_build')
