import unittest
from onmt.modules.copy_generator import CopyGenerator, CopyGeneratorLoss

import itertools
from copy import deepcopy

import torch
from torch.nn.functional import softmax

from onmt.tests.utils_for_tests import product_dict


class TestCopyGenerator(unittest.TestCase):
    INIT_CASES = list(product_dict(
        input_size=[172],
        output_size=[319],
        pad_idx=[0, 39],
    ))
    PARAMS = list(product_dict(
        batch_size=[1, 14],
        max_seq_len=[23],
        tgt_max_len=[50],
        n_extra_words=[107]
    ))

    @classmethod
    def dummy_inputs(cls, params, init_case):
        hidden = torch.randn((params["batch_size"] * params["tgt_max_len"],
                              init_case["input_size"]))
        attn = torch.randn((params["batch_size"] * params["tgt_max_len"],
                            params["max_seq_len"]))
        src_map = torch.randn((params["max_seq_len"], params["batch_size"],
                               params["n_extra_words"]))
        return hidden, attn, src_map

    @classmethod
    def expected_shape(cls, params, init_case):
        return params["tgt_max_len"] * params["batch_size"], \
               init_case["output_size"] + params["n_extra_words"]

    def test_copy_gen_forward_shape(self):
        for params, init_case in itertools.product(
                self.PARAMS, self.INIT_CASES):
            cgen = CopyGenerator(**init_case)
            dummy_in = self.dummy_inputs(params, init_case)
            res = cgen(*dummy_in)
            expected_shape = self.expected_shape(params, init_case)
            self.assertEqual(res.shape, expected_shape, init_case.__str__())

    def test_copy_gen_outp_has_no_prob_of_pad(self):
        for params, init_case in itertools.product(
                self.PARAMS, self.INIT_CASES):
            cgen = CopyGenerator(**init_case)
            dummy_in = self.dummy_inputs(params, init_case)
            res = cgen(*dummy_in)
            self.assertTrue(
                res[:, init_case["pad_idx"]].allclose(torch.tensor(0.0)))

    def test_copy_gen_trainable_params_update(self):
        for params, init_case in itertools.product(
                self.PARAMS, self.INIT_CASES):
            cgen = CopyGenerator(**init_case)
            trainable_params = {n: p for n, p in cgen.named_parameters()
                                if p.requires_grad}
            assert len(trainable_params) > 0  # sanity check
            old_weights = deepcopy(trainable_params)
            dummy_in = self.dummy_inputs(params, init_case)
            res = cgen(*dummy_in)
            pretend_loss = res.sum()
            pretend_loss.backward()
            dummy_optim = torch.optim.SGD(trainable_params.values(), 1)
            dummy_optim.step()
            for param_name in old_weights.keys():
                self.assertTrue(
                    trainable_params[param_name]
                    .ne(old_weights[param_name]).any(),
                    param_name + " " + init_case.__str__())


class TestCopyGeneratorLoss(unittest.TestCase):
    INIT_CASES = list(product_dict(
        vocab_size=[172],
        unk_index=[0, 39],
        ignore_index=[1, 17],  # pad idx
        force_copy=[True, False]
    ))
    PARAMS = list(product_dict(
        batch_size=[1, 14],
        tgt_max_len=[50],
        n_extra_words=[107]
    ))

    @classmethod
    def dummy_inputs(cls, params, init_case):
        n_unique_src_words = 13
        scores = torch.randn((params["batch_size"] * params["tgt_max_len"],
                              init_case["vocab_size"] + n_unique_src_words))
        scores = softmax(scores, dim=1)
        align = torch.randint(0, n_unique_src_words,
                              (params["batch_size"] * params["tgt_max_len"],))
        target = torch.randint(0, init_case["vocab_size"],
                               (params["batch_size"] * params["tgt_max_len"],))
        target[0] = init_case["unk_index"]
        target[1] = init_case["ignore_index"]
        return scores, align, target

    @classmethod
    def expected_shape(cls, params, init_case):
        return (params["batch_size"] * params["tgt_max_len"],)

    def test_copy_loss_forward_shape(self):
        for params, init_case in itertools.product(
                self.PARAMS, self.INIT_CASES):
            loss = CopyGeneratorLoss(**init_case)
            dummy_in = self.dummy_inputs(params, init_case)
            res = loss(*dummy_in)
            expected_shape = self.expected_shape(params, init_case)
            self.assertEqual(res.shape, expected_shape, init_case.__str__())

    def test_copy_loss_ignore_index_is_ignored(self):
        for params, init_case in itertools.product(
                self.PARAMS, self.INIT_CASES):
            loss = CopyGeneratorLoss(**init_case)
            scores, align, target = self.dummy_inputs(params, init_case)
            res = loss(scores, align, target)
            should_be_ignored = (target == init_case["ignore_index"]).nonzero(
                as_tuple=False)
            assert len(should_be_ignored) > 0  # otherwise not testing anything
            self.assertTrue(res[should_be_ignored].allclose(torch.tensor(0.0)))

    def test_copy_loss_output_range_is_positive(self):
        for params, init_case in itertools.product(
                self.PARAMS, self.INIT_CASES):
            loss = CopyGeneratorLoss(**init_case)
            dummy_in = self.dummy_inputs(params, init_case)
            res = loss(*dummy_in)
            self.assertTrue((res >= 0).all())
