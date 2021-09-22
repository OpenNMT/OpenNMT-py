"""Here come the tests for implemented transform."""
import unittest

import copy
import yaml
import math
from argparse import Namespace
from onmt.transforms import (
    get_transforms_cls,
    get_specials,
    make_transforms,
    TransformPipe,
)
from onmt.transforms.bart import BARTNoising


class TestTransform(unittest.TestCase):
    def test_transform_register(self):
        builtin_transform = [
            "filtertoolong",
            "prefix",
            "sentencepiece",
            "bpe",
            "onmt_tokenize",
            "bart",
            "switchout",
            "tokendrop",
            "tokenmask",
        ]
        get_transforms_cls(builtin_transform)

    def test_vocab_required_transform(self):
        transforms_cls = get_transforms_cls(["bart", "switchout"])
        opt = Namespace(seed=-1, switchout_temperature=1.0)
        # transforms that require vocab will not create if not provide vocab
        transforms = make_transforms(opt, transforms_cls, fields=None)
        self.assertEqual(len(transforms), 0)
        with self.assertRaises(ValueError):
            transforms_cls["switchout"](opt).warm_up(vocabs=None)
            transforms_cls["bart"](opt).warm_up(vocabs=None)

    def test_transform_specials(self):
        transforms_cls = get_transforms_cls(["prefix"])
        corpora = yaml.safe_load("""
            trainset:
                path_src: data/src-train.txt
                path_tgt: data/tgt-train.txt
                transforms: ["prefix"]
                weight: 1
                src_prefix: "｟_pf_src｠"
                tgt_prefix: "｟_pf_tgt｠"
        """)
        opt = Namespace(data=corpora)
        specials = get_specials(opt, transforms_cls)
        specials_expected = {"src": {"｟_pf_src｠"}, "tgt": {"｟_pf_tgt｠"}}
        self.assertEqual(specials, specials_expected)

    def test_transform_pipe(self):
        # 1. Init first transform in the pipe
        prefix_cls = get_transforms_cls(["prefix"])["prefix"]
        corpora = yaml.safe_load("""
            trainset:
                path_src: data/src-train.txt
                path_tgt: data/tgt-train.txt
                transforms: [prefix, filtertoolong]
                weight: 1
                src_prefix: "｟_pf_src｠"
                tgt_prefix: "｟_pf_tgt｠"
        """)
        opt = Namespace(data=corpora, seed=-1)
        prefix_transform = prefix_cls(opt)
        prefix_transform.warm_up()
        # 2. Init second transform in the pipe
        filter_cls = get_transforms_cls(["filtertoolong"])["filtertoolong"]
        opt = Namespace(src_seq_length=4, tgt_seq_length=4)
        filter_transform = filter_cls(opt)
        # 3. Sequential combine them into a transform pipe
        transform_pipe = TransformPipe.build_from(
            [prefix_transform, filter_transform]
        )
        ex = {
            "src": ["Hello", ",", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        # 4. apply transform pipe for example
        ex_after = transform_pipe.apply(
            copy.deepcopy(ex), corpus_name="trainset"
        )
        # 5. example after the pipe exceed the length limit, thus filtered
        self.assertIsNone(ex_after)
        # 6. Transform statistics registed (here for filtertoolong)
        self.assertTrue(len(transform_pipe.statistics.observables) > 0)
        msg = transform_pipe.statistics.report()
        self.assertIsNotNone(msg)
        # 7. after report, statistics become empty as a fresh start
        self.assertTrue(len(transform_pipe.statistics.observables) == 0)


class TestMiscTransform(unittest.TestCase):
    def test_prefix(self):
        prefix_cls = get_transforms_cls(["prefix"])["prefix"]
        corpora = yaml.safe_load("""
            trainset:
                path_src: data/src-train.txt
                path_tgt: data/tgt-train.txt
                transforms: [prefix]
                weight: 1
                src_prefix: "｟_pf_src｠"
                tgt_prefix: "｟_pf_tgt｠"
        """)
        opt = Namespace(data=corpora, seed=-1)
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
        opt = Namespace(src_seq_length=100, tgt_seq_length=100)
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
            "src_subword_model": "data/sample.bpe",
            "tgt_subword_model": "data/sample.bpe",
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
            "src": ["H@@", "ell@@", "o", "world", "."],
            "tgt": ["B@@", "on@@", "j@@", "our", "le", "mon@@", "de", "."],
        }
        self.assertEqual(ex, ex_gold)
        # test BPE-dropout:
        bpe_transform.dropout["src"] = 1.0
        tokens = ["Another", "world", "."]
        gold_bpe = ["A@@", "no@@", "ther", "world", "."]
        gold_dropout = [
            "A@@", "n@@", "o@@", "t@@", "h@@", "e@@", "r",
            "w@@", "o@@", "r@@", "l@@", "d", ".",
        ]
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
            "src": ["▁H", "el", "lo", "▁world", "▁."],
            "tgt": ["▁B", "on", "j", "o", "ur", "▁le", "▁m", "on", "de", "▁."],
        }
        self.assertEqual(ex, ex_gold)
        # test SP regularization:
        sp_transform.src_subword_nbest = 4
        tokens = ["Another", "world", "."]
        gold_sp = ["▁An", "other", "▁world", "▁."]
        # 1. enable regularization for training example
        after_sp = sp_transform._tokenize(tokens, is_train=True)
        self.assertEqual(after_sp, ["▁An", "o", "ther", "▁world", "▁."])
        # 2. disable regularization for not training example
        after_sp = sp_transform._tokenize(tokens, is_train=False)
        self.assertEqual(after_sp, gold_sp)

    def test_pyonmttok_bpe(self):
        onmttok_cls = get_transforms_cls(["onmt_tokenize"])["onmt_tokenize"]
        base_opt = copy.copy(self.base_opts)
        base_opt["src_subword_type"] = "bpe"
        base_opt["tgt_subword_type"] = "bpe"
        onmt_args = "{'mode': 'space', 'joiner_annotate': True}"
        base_opt["src_onmttok_kwargs"] = onmt_args
        base_opt["tgt_onmttok_kwargs"] = onmt_args
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
            "src": ["H￭", "ell￭", "o", "world", "."],
            "tgt": ["B￭", "on￭", "j￭", "our", "le", "mon￭", "de", "."],
        }
        self.assertEqual(ex, ex_gold)

    def test_pyonmttok_sp(self):
        onmttok_cls = get_transforms_cls(["onmt_tokenize"])["onmt_tokenize"]
        base_opt = copy.copy(self.base_opts)
        base_opt["src_subword_type"] = "sentencepiece"
        base_opt["tgt_subword_type"] = "sentencepiece"
        base_opt["src_subword_model"] = "data/sample.sp.model"
        base_opt["tgt_subword_model"] = "data/sample.sp.model"
        onmt_args = "{'mode': 'none', 'spacer_annotate': True}"
        base_opt["src_onmttok_kwargs"] = onmt_args
        base_opt["tgt_onmttok_kwargs"] = onmt_args
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
            "src": ["▁H", "el", "lo", "▁world", "▁."],
            "tgt": ["▁B", "on", "j", "o", "ur", "▁le", "▁m", "on", "de", "▁."],
        }
        self.assertEqual(ex, ex_gold)


class TestSamplingTransform(unittest.TestCase):
    def test_tokendrop(self):
        tokendrop_cls = get_transforms_cls(["tokendrop"])["tokendrop"]
        opt = Namespace(seed=3434, tokendrop_temperature=0.1)
        tokendrop_transform = tokendrop_cls(opt)
        tokendrop_transform.warm_up()
        ex = {
            "src": ["Hello", ",", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        # Not apply token drop for not training example
        ex_after = tokendrop_transform.apply(copy.deepcopy(ex), is_train=False)
        self.assertEqual(ex_after, ex)
        # apply token drop for training example
        ex_after = tokendrop_transform.apply(copy.deepcopy(ex), is_train=True)
        self.assertNotEqual(ex_after, ex)

    def test_tokenmask(self):
        tokenmask_cls = get_transforms_cls(["tokenmask"])["tokenmask"]
        opt = Namespace(seed=3434, tokenmask_temperature=0.1)
        tokenmask_transform = tokenmask_cls(opt)
        tokenmask_transform.warm_up()
        ex = {
            "src": ["Hello", ",", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        # Not apply token mask for not training example
        ex_after = tokenmask_transform.apply(copy.deepcopy(ex), is_train=False)
        self.assertEqual(ex_after, ex)
        # apply token mask for training example
        ex_after = tokenmask_transform.apply(copy.deepcopy(ex), is_train=True)
        self.assertNotEqual(ex_after, ex)

    def test_switchout(self):
        switchout_cls = get_transforms_cls(["switchout"])["switchout"]
        opt = Namespace(seed=3434, switchout_temperature=0.1)
        switchout_transform = switchout_cls(opt)
        with self.assertRaises(ValueError):
            # require vocabs to warm_up
            switchout_transform.warm_up(vocabs=None)
        vocabs = {
            "src": Namespace(itos=["A", "Fake", "vocab"]),
            "tgt": Namespace(itos=["A", "Fake", "vocab"]),
        }
        switchout_transform.warm_up(vocabs=vocabs)
        ex = {
            "src": ["Hello", ",", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        # Not apply token mask for not training example
        ex_after = switchout_transform.apply(copy.deepcopy(ex), is_train=False)
        self.assertEqual(ex_after, ex)
        # apply token mask for training example
        ex_after = switchout_transform.apply(copy.deepcopy(ex), is_train=True)
        self.assertNotEqual(ex_after, ex)


class TestBARTNoising(unittest.TestCase):
    def setUp(self):
        BARTNoising.set_random_seed(1234)
        self.MASK_TOK = "[MASK]"
        self.FAKE_VOCAB = "[TESTING]"

    def test_sentence_permute(self):
        sent1 = ["Hello", "world", "."]
        sent2 = ["Sentence", "1", "!"]
        sent3 = ["Sentence", "2", "!"]
        sent4 = ["Sentence", "3", "!"]

        bart_noise = BARTNoising(
            vocab=[self.FAKE_VOCAB],
            permute_sent_ratio=0.5,
            replace_length=0,  # not raise Error
            # Defalt: full_stop_token=[".", "?", "!"]
        )
        tokens = sent1 + sent2 + sent3 + sent4
        ends = bart_noise._get_sentence_borders(tokens).tolist()
        self.assertEqual(ends, [3, 6, 9, 12])
        tokens_perm = bart_noise.apply(tokens)
        expected_tokens = sent2 + sent1 + sent3 + sent4
        self.assertEqual(expected_tokens, tokens_perm)

    def test_rotate(self):
        bart_noise = BARTNoising(
            vocab=[self.FAKE_VOCAB],
            rotate_ratio=1.0,
            replace_length=0,  # not raise Error
        )
        tokens = ["This", "looks", "really", "good", "!"]
        rotated = bart_noise.apply(tokens)
        self.assertNotEqual(tokens, rotated)
        not_rotate = bart_noise.rolling_noise(tokens, p=0.0)
        self.assertEqual(tokens, not_rotate)

    def test_token_insert(self):
        bart_noise = BARTNoising(
            vocab=[self.FAKE_VOCAB],
            mask_tok=self.MASK_TOK,
            insert_ratio=0.5,
            random_ratio=0.3,
            replace_length=0,  # not raise Error
            # Defalt: full_stop_token=[".", "?", "!"]
        )
        tokens = ["This", "looks", "really", "good", "!"]
        inserted = bart_noise.apply(tokens)
        n_insert = math.ceil(len(tokens) * bart_noise.insert_ratio)
        inserted_len = n_insert + len(tokens)
        self.assertEqual(len(inserted), inserted_len)
        # random_ratio of inserted tokens are chosen in vocab
        n_random = math.ceil(n_insert * bart_noise.random_ratio)
        self.assertEqual(
            sum(1 if tok == self.FAKE_VOCAB else 0 for tok in inserted),
            n_random,
        )
        # others are MASK_TOK
        self.assertEqual(
            sum(1 if tok == self.MASK_TOK else 0 for tok in inserted),
            n_insert - n_random,
        )

    def test_token_mask(self):
        """Mask will be done on token level.

        Condition:
        * `mask_length` == subword;
        * or not specify subword marker (joiner/spacer) by `is_joiner`.
        """
        bart_noise = BARTNoising(
            vocab=[self.FAKE_VOCAB],
            mask_tok=self.MASK_TOK,
            mask_ratio=0.5,
            mask_length="subword",
            replace_length=0,  # 0 to drop them, 1 to replace them with MASK
            # insert_ratio=0.0,
            # random_ratio=0.0,
            # Defalt: full_stop_token=[".", "?", "!"]
        )
        tokens = ["H￭", "ell￭", "o", "world", "."]
        # all token are considered as an individual word
        self.assertTrue(all(bart_noise._is_word_start(tokens)))
        n_tokens = len(tokens)

        # 1. tokens are dropped when replace_length is 0
        masked = bart_noise.apply(tokens)
        n_masked = math.ceil(n_tokens * bart_noise.mask_ratio)
        # print(f"token delete: {masked} / {tokens}")
        self.assertEqual(len(masked), n_tokens - n_masked)

        # 2. tokens are replaced by MASK when replace_length is 1
        bart_noise.replace_length = 1
        masked = bart_noise.apply(tokens)
        n_masked = math.ceil(n_tokens * bart_noise.mask_ratio)
        # print(f"token mask: {masked} / {tokens}")
        self.assertEqual(len(masked), n_tokens)
        self.assertEqual(
            sum([1 if tok == self.MASK_TOK else 0 for tok in masked]), n_masked
        )

    def test_whole_word_mask(self):
        """Mask will be done on whole word that may across multiply token.

        Condition:
        * `mask_length` == word;
        * specify subword marker in order to find word boundary.
        """
        bart_noise = BARTNoising(
            vocab=[self.FAKE_VOCAB],
            mask_tok=self.MASK_TOK,
            mask_ratio=0.5,
            mask_length="word",
            is_joiner=True,
            replace_length=0,  # 0 to drop them, 1 to replace them with MASK
            # insert_ratio=0.0,
            # random_ratio=0.0,
            # Defalt: full_stop_token=[".", "?", "!"]
        )
        tokens = ["H￭", "ell￭", "o", "wor￭", "ld", "."]
        # start token of word are identified using subword marker
        token_starts = [True, False, False, True, False, True]
        self.assertEqual(bart_noise._is_word_start(tokens), token_starts)

        # 1. replace_length 0: "words" are dropped
        masked = bart_noise.apply(copy.copy(tokens))
        n_words = sum(token_starts)
        n_masked = math.ceil(n_words * bart_noise.mask_ratio)
        # print(f"word delete: {masked} / {tokens}")
        # self.assertEqual(len(masked), n_words - n_masked)

        # 2. replace_length 1: "words" are replaced with a single MASK
        bart_noise.replace_length = 1
        masked = bart_noise.apply(copy.copy(tokens))
        # print(f"whole word single mask: {masked} / {tokens}")
        # len(masked) depend on number of tokens in select word
        n_words = sum(token_starts)
        n_masked = math.ceil(n_words * bart_noise.mask_ratio)
        self.assertEqual(
            sum(1 if tok == self.MASK_TOK else 0 for tok in masked), n_masked
        )

        # 3. replace_length -1: all tokens in "words" are replaced with MASK
        bart_noise.replace_length = -1
        masked = bart_noise.apply(copy.copy(tokens))
        # print(f"whole word multi mask: {masked} / {tokens}")
        self.assertEqual(len(masked), len(tokens))  # length won't change
        n_words = sum(token_starts)
        n_masked = math.ceil(n_words * bart_noise.mask_ratio)
        # number of mask_tok depend on number of tokens in selected word
        # number of MASK_TOK can be greater than n_masked
        self.assertTrue(
            sum(1 if tok == self.MASK_TOK else 0 for tok in masked) > n_masked
        )

    def test_span_infilling(self):
        bart_noise = BARTNoising(
            vocab=[self.FAKE_VOCAB],
            mask_tok=self.MASK_TOK,
            mask_ratio=0.5,
            mask_length="span-poisson",
            poisson_lambda=3.0,
            is_joiner=True,
            replace_length=1,
            # insert_ratio=0.5,
            # random_ratio=0.3,
            # Defalt: full_stop_token=[".", "?", "!"]
        )
        self.assertIsNotNone(bart_noise.mask_span_distribution)
        tokens = ["H￭", "ell￭", "o", "world", ".", "An￭", "other", "!"]
        # start token of word are identified using subword marker
        token_starts = [True, False, False, True, True, True, False, True]
        self.assertEqual(bart_noise._is_word_start(tokens), token_starts)
        bart_noise.apply(copy.copy(tokens))
        # n_words = sum(token_starts)
        # n_masked = math.ceil(n_words * bart_noise.mask_ratio)
        # print(f"Text Span Infilling: {infillied} / {tokens}")
        # print(n_words, n_masked)


class TestFeaturesTransform(unittest.TestCase):
    def test_inferfeats(self):
        inferfeats_cls = get_transforms_cls(["inferfeats"])["inferfeats"]
        opt = Namespace(
            reversible_tokenization="joiner",
            prior_tokenization=False)
        inferfeats_transform = inferfeats_cls(opt)

        ex_in = {
            "src": ['however', '￭,', 'according', 'to', 'the', 'logs',
                    '￭,', 'she', 'is', 'hard', '￭-￭', 'working', '￭.'],
            "tgt": ['however', '￭,', 'according', 'to', 'the', 'logs',
                    '￭,', 'she', 'is', 'hard', '￭-￭', 'working', '￭.']
        }
        ex_out = inferfeats_transform.apply(ex_in)
        self.assertIs(ex_out, ex_in)

        ex_in["src_feats"] = {
            "feat_0": ["A", "A", "A", "A", "B", "A", "A", "C"]
        }
        ex_out = inferfeats_transform.apply(ex_in)
        self.assertEqual(
            ex_out["src_feats"]["feat_0"],
            ["A", "<null>", "A", "A", "A", "B", "<null>", "A",
             "A", "C", "<null>", "C", "<null>"])

        ex_in["src"] = ['｟mrk_case_modifier_C｠', 'however', '￭,',
                        'according', 'to', 'the', 'logs', '￭,',
                        '｟mrk_begin_case_region_U｠', 'she', 'is', 'hard',
                        '￭-￭', 'working', '｟mrk_end_case_region_U｠', '￭.']
        ex_in["src_feats"] = {
            "feat_0": ["A", "A", "A", "A", "B", "A", "A", "C"]
        }
        ex_out = inferfeats_transform.apply(ex_in)
        self.assertEqual(
            ex_out["src_feats"]["feat_0"],
            ["A", "A", "<null>", "A", "A", "A", "B", "<null>",
             "A", "A", "A", "C", "<null>", "C", "C", "<null>"])
