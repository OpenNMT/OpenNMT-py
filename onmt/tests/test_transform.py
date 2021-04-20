"""Here come the tests for implemented transform."""
import unittest

import os
import copy
import yaml
from argparse import Namespace
from onmt.transforms import get_transforms_cls, get_specials, make_transforms


class TestTransform(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_corpora = yaml.safe_load('''
            trainset:
                path_src: data/src-train.txt
                path_tgt: data/tgt-train.txt
                weight: 1
        ''')

    def test_transform_register(self):
        builtin_transform = [
            'filtertoolong',
            'prefix',
            'sentencepiece',
            'bpe',
            'onmt_tokenize',
            'bart',
            'switchout',
            'tokendrop',
            'tokenmask'
        ]
        get_transforms_cls(builtin_transform)

    def test_vocab_required_transform(self):
        transforms_cls = get_transforms_cls(["bart", "switchout"])
        corpora = copy.deepcopy(self.base_corpora)
        corpora["trainset"]["transforms"] = ["bart", "switchout"]
        opt = Namespace(data=corpora, seed=-1, switchout_temperature=1.0)
        # transforms that require vocab will not create if not provide vocab
        transforms = make_transforms(opt, transforms_cls, fields=None)
        self.assertEqual(len(transforms), 0)
        with self.assertRaises(ValueError):
            transforms_cls["switchout"](opt).warm_up(vocabs=None)
            transforms_cls["bart"](opt).warm_up(vocabs=None)

    def test_transform_specials(self):
        transforms_cls = get_transforms_cls(["prefix"])
        corpora = copy.deepcopy(self.base_corpora)
        corpora["trainset"]["transforms"] = ["prefix"]
        corpora["trainset"]["src_prefix"] = "｟_pf_src｠"
        corpora["trainset"]["tgt_prefix"] = "｟_pf_tgt｠"
        opt = Namespace(data=corpora)
        specials = get_specials(opt, transforms_cls)
        specials_expected = {
            "src": {"｟_pf_src｠"},
            "tgt": {"｟_pf_tgt｠"}
        }
        self.assertEqual(specials, specials_expected)


class TestMiscTransform(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_corpora = yaml.safe_load('''
            trainset:
                path_src: data/src-train.txt
                path_tgt: data/tgt-train.txt
                transforms: [prefix, filtertoolong]
                weight: 1
                src_prefix: "｟_pf_src｠"
                tgt_prefix: "｟_pf_tgt｠"
        ''')

    def test_prefix(self):
        prefix_cls = get_transforms_cls(["prefix"])["prefix"]
        opt = Namespace(data=self.base_corpora, seed=-1)
        prefix_transform = prefix_cls(opt)
        prefix_transform.warm_up()
        self.assertIn("trainset", prefix_transform.prefix_dict)

        ex_in = {
            "src": ["Hello", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        with self.assertRaises(ValueError):
            prefix_transform.apply(ex_in)
            prefix_transform.apply(ex_in, corpus_name="validset")
        ex_out = prefix_transform.apply(ex_in, corpus_name="trainset")
        self.assertEqual(ex_out["src"][0], "｟_pf_src｠")
        self.assertEqual(ex_out["tgt"][0], "｟_pf_tgt｠")

    def test_filter_too_long(self):
        filter_cls = get_transforms_cls(["filtertoolong"])["filtertoolong"]
        opt = Namespace(
            data=self.base_corpora,
            src_seq_length=100,
            tgt_seq_length=100
        )
        filter_transform = filter_cls(opt)
        # filter_transform.warm_up()
        ex_in = {
            "src": ["Hello", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        ex_out = filter_transform.apply(ex_in)
        self.assertIs(ex_out, ex_in)
        filter_transform.tgt_seq_length = 2
        ex_out = filter_transform.apply(ex_in)
        self.assertIsNone(ex_out)


class TestSubwordTransform(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_opts = {
            "seed": 3431,
            "share_vocab": False,
            "src_subword_model" : "data/sample.bpe",
            "tgt_subword_model" : "data/sample.bpe",
            "src_subword_nbest": 1,
            "tgt_subword_nbest": 1,
            "src_subword_alpha": 0.0,
            "tgt_subword_alpha": 0.0,
            "src_subword_vocab": "",
            "tgt_subword_vocab": "",
            "src_vocab_threshold": 0,
            "tgt_vocab_threshold": 0,
        }

    def test_bpe(self):
        bpe_cls = get_transforms_cls(["bpe"])["bpe"]
        opt = Namespace(**self.base_opts)
        bpe_cls._validate_options(opt)
        bpe_transform = bpe_cls(opt)
        bpe_transform.warm_up()
        ex = {
            "src": ["Hello", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        bpe_transform.apply(ex, is_train=True)
        ex_gold = {
            "src": ['H@@', 'ell@@', 'o', 'world', '.'],
            "tgt": ['B@@', 'on@@', 'j@@', 'our', 'le', 'mon@@', 'de', '.']
        }
        self.assertEqual(ex, ex_gold)
        # test BPE-dropout:
        bpe_transform.dropout['src'] = 1.0
        tokens = ["Another", "world", "."]
        gold_bpe = ['A@@', 'no@@', 'ther', 'world', '.']
        gold_dropout = ['A@@', 'n@@', 'o@@', 't@@', 'h@@', 'e@@', 'r', 'w@@', 'o@@', 'r@@', 'l@@', 'd', '.']
        # 1. disable bpe dropout for not training example
        after_bpe = bpe_transform._tokenize(tokens, is_train=False)
        self.assertEqual(after_bpe, gold_bpe)
        # 2. enable bpe dropout for training example
        after_bpe = bpe_transform._tokenize(tokens, is_train=True)
        self.assertEqual(after_bpe, gold_dropout)
        # 3. (NOTE) disable dropout won't take effect if already seen
        # this is caused by the cache mechanism in bpe:
        # return cached subword if the original token is seen when no dropout
        after_bpe2 = bpe_transform._tokenize(tokens, is_train=False)
        self.assertEqual(after_bpe2, gold_dropout)

    def test_sentencepiece(self):
        sp_cls = get_transforms_cls(["sentencepiece"])["sentencepiece"]
        base_opt = copy.copy(self.base_opts)
        base_opt["src_subword_model"] = "data/sample.sp.model"
        base_opt["tgt_subword_model"] = "data/sample.sp.model"
        opt = Namespace(**base_opt)
        sp_cls._validate_options(opt)
        sp_transform = sp_cls(opt)
        sp_transform.warm_up()
        ex = {
            "src": ["Hello", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        sp_transform.apply(ex, is_train=True)
        ex_gold = {
            "src": ['▁H', 'el', 'lo', '▁world', '▁.'],
            "tgt": ['▁B', 'on', 'j', 'o', 'ur', '▁le', '▁m', 'on', 'de', '▁.']
        }
        self.assertEqual(ex, ex_gold)
        # test SP regularization:
        sp_transform.src_subword_nbest = 4
        tokens = ["Another", "world", "."]
        gold_sp = ['▁An', 'other', '▁world', '▁.']
        # 1. enable regularization for training example
        after_sp = sp_transform._tokenize(tokens, is_train=True)
        self.assertEqual(after_sp, ['▁An', 'o', 'ther', '▁world', '▁.'])
        # 2. disable regularization for not training example
        after_sp = sp_transform._tokenize(tokens, is_train=False)
        self.assertEqual(after_sp, gold_sp)

    def test_pyonmttok_bpe(self):
        onmttok_cls = get_transforms_cls(["onmt_tokenize"])["onmt_tokenize"]
        base_opt = copy.copy(self.base_opts)
        base_opt["src_subword_type"] = "bpe"
        base_opt["tgt_subword_type"] = "bpe"
        base_opt["src_onmttok_kwargs"] = "{'mode': 'space', 'joiner_annotate': True}"
        base_opt["tgt_onmttok_kwargs"] = "{'mode': 'space', 'joiner_annotate': True}"
        opt = Namespace(**base_opt)
        onmttok_cls._validate_options(opt)
        onmttok_transform = onmttok_cls(opt)
        onmttok_transform.warm_up()
        ex = {
            "src": ["Hello", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        onmttok_transform.apply(ex, is_train=True)
        ex_gold = {
            "src": ['H￭', 'ell￭', 'o', 'world', '.'],
            "tgt": ['B￭', 'on￭', 'j￭', 'our', 'le', 'mon￭', 'de', '.']
        }
        self.assertEqual(ex, ex_gold)

    def test_pyonmttok_sp(self):
        onmttok_cls = get_transforms_cls(["onmt_tokenize"])["onmt_tokenize"]
        base_opt = copy.copy(self.base_opts)
        base_opt["src_subword_type"] = "sentencepiece"
        base_opt["tgt_subword_type"] = "sentencepiece"
        base_opt["src_subword_model"] = "data/sample.sp.model"
        base_opt["tgt_subword_model"] = "data/sample.sp.model"
        base_opt["src_onmttok_kwargs"] = "{'mode': 'none', 'spacer_annotate': True}"
        base_opt["tgt_onmttok_kwargs"] = "{'mode': 'none', 'spacer_annotate': True}"
        opt = Namespace(**base_opt)
        onmttok_cls._validate_options(opt)
        onmttok_transform = onmttok_cls(opt)
        onmttok_transform.warm_up()
        ex = {
            "src": ["Hello", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        onmttok_transform.apply(ex, is_train=True)
        ex_gold = {
            "src": ['▁H', 'el', 'lo', '▁world', '▁.'],
            "tgt": ['▁B', 'on', 'j', 'o', 'ur', '▁le', '▁m', 'on', 'de', '▁.']
        }
        self.assertEqual(ex, ex_gold)
