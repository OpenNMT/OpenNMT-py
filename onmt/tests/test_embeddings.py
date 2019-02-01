import unittest
from onmt.modules.embeddings import Embeddings

import itertools
from copy import deepcopy

import torch

from onmt.tests.utils_for_tests import product_dict


class TestEmbeddings(unittest.TestCase):
    INIT_CASES = list(product_dict(
        word_vec_size=[172],
        word_vocab_size=[319],
        word_padding_idx=[17],
        position_encoding=[False, True],
        feat_merge=["first", "concat", "sum", "mlp"],
        feat_vec_exponent=[-1, 1.1, 0.7],
        feat_vec_size=[0, 200],
        feat_padding_idx=[[], [29], [0, 1]],
        feat_vocab_sizes=[[], [39], [401, 39]],
        dropout=[0, 0.5],
        fix_word_vecs=[False, True]
    ))
    PARAMS = list(product_dict(
        batch_size=[1, 14],
        max_seq_len=[23]
    ))

    @classmethod
    def case_is_degenerate(cls, case):
        no_feats = len(case["feat_vocab_sizes"]) == 0
        if case["feat_merge"] != "first" and no_feats:
            return True
        if case["feat_merge"] == "first" and not no_feats:
            return True
        if case["feat_merge"] == "concat" and case["feat_vec_exponent"] != -1:
            return True
        if no_feats and case["feat_vec_exponent"] != -1:
            return True
        if len(case["feat_vocab_sizes"]) != len(case["feat_padding_idx"]):
            return True
        if case["feat_vec_size"] == 0 and case["feat_vec_exponent"] <= 0:
            return True
        if case["feat_merge"] == "sum":
            if case["feat_vec_exponent"] != -1:
                return True
            if case["feat_vec_size"] != 0:
                return True
        if case["feat_vec_size"] != 0 and case["feat_vec_exponent"] != -1:
            return True
        return False

    @classmethod
    def cases(cls):
        for case in cls.INIT_CASES:
            if not cls.case_is_degenerate(case):
                yield case

    @classmethod
    def dummy_inputs(cls, params, init_case):
        max_seq_len = params["max_seq_len"]
        batch_size = params["batch_size"]
        fv_sizes = init_case["feat_vocab_sizes"]
        n_words = init_case["word_vocab_size"]
        voc_sizes = [n_words] + fv_sizes
        pad_idxs = [init_case["word_padding_idx"]] + \
            init_case["feat_padding_idx"]
        lengths = torch.randint(0, max_seq_len, (batch_size,))
        lengths[0] = max_seq_len
        inps = torch.empty((max_seq_len, batch_size, len(voc_sizes)),
                           dtype=torch.long)
        for f, (voc_size, pad_idx) in enumerate(zip(voc_sizes, pad_idxs)):
            for b, len_ in enumerate(lengths):
                inps[:len_, b, f] = torch.randint(0, voc_size-1, (len_,))
                inps[len_:, b, f] = pad_idx
        return inps

    @classmethod
    def expected_shape(cls, params, init_case):
        wvs = init_case["word_vec_size"]
        fvs = init_case["feat_vec_size"]
        nf = len(init_case["feat_vocab_sizes"])
        size = wvs
        if init_case["feat_merge"] not in {"sum", "mlp"}:
            size += nf * fvs
        return params["max_seq_len"], params["batch_size"], size

    def test_embeddings_forward_shape(self):
        for params, init_case in itertools.product(self.PARAMS, self.cases()):
            emb = Embeddings(**init_case)
            dummy_in = self.dummy_inputs(params, init_case)
            res = emb(dummy_in)
            expected_shape = self.expected_shape(params, init_case)
            self.assertEqual(res.shape, expected_shape, init_case.__str__())

    def test_embeddings_trainable_params(self):
        for params, init_case in itertools.product(self.PARAMS,
                                                   self.cases()):
            emb = Embeddings(**init_case)
            trainable_params = {n: p for n, p in emb.named_parameters()
                                if p.requires_grad}
            # first check there's nothing unexpectedly not trainable
            for key in emb.state_dict():
                if key not in trainable_params:
                    if key.endswith("emb_luts.0.weight") and \
                            init_case["fix_word_vecs"]:
                        # ok: word embeddings shouldn't be trainable
                        # if word vecs are fixed
                        continue
                    if key.endswith(".pe.pe"):
                        # ok: positional encodings shouldn't be trainable
                        assert init_case["position_encoding"]
                        continue
                    else:
                        self.fail("Param {:s} is unexpectedly not "
                                  "trainable.".format(key))
            # then check nothing unexpectedly trainable
            if init_case["fix_word_vecs"]:
                self.assertFalse(
                    any(trainable_param.endswith("emb_luts.0.weight")
                        for trainable_param in trainable_params),
                    "Word embedding is trainable but word vecs are fixed.")
            if init_case["position_encoding"]:
                self.assertFalse(
                    any(trainable_p.endswith(".pe.pe")
                        for trainable_p in trainable_params),
                    "Positional encoding is trainable.")

    def test_embeddings_trainable_params_update(self):
        for params, init_case in itertools.product(self.PARAMS, self.cases()):
            emb = Embeddings(**init_case)
            trainable_params = {n: p for n, p in emb.named_parameters()
                                if p.requires_grad}
            if len(trainable_params) > 0:
                old_weights = deepcopy(trainable_params)
                dummy_in = self.dummy_inputs(params, init_case)
                res = emb(dummy_in)
                pretend_loss = res.sum()
                pretend_loss.backward()
                dummy_optim = torch.optim.SGD(trainable_params.values(), 1)
                dummy_optim.step()
                for param_name in old_weights.keys():
                    self.assertTrue(
                        trainable_params[param_name]
                        .ne(old_weights[param_name]).any(),
                        param_name + " " + init_case.__str__())
