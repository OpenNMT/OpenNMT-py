import unittest
from onmt.inputters.text_dataset import TextMultiField

import itertools

from torchtext.data import Field


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


class TestTextMultiField(unittest.TestCase):
    INIT_CASES = list(product_dict(
        base_name=["base_field", "zbase_field"],
        base_field=[Field()],
        feats_fields=[
            [],
            [("a", Field(sequential=False))],
            [("r", Field(sequential=False)), ("b", Field(sequential=False))]]
    ))

    # def test_process_shape(self):
    #     for init_case in self.INIT_CASES:
    #         mf = TextMultiField(**init_case)

    def test_preprocess_shape(self):
        for init_case in self.INIT_CASES:
            mf = TextMultiField(**init_case)
            sample_str = "dummy input here ."
            proc = mf.preprocess(sample_str)
            self.assertEqual(len(proc), len(init_case["feats_fields"]) + 1)

    def test_base_field(self):
        for init_case in self.INIT_CASES:
            mf = TextMultiField(**init_case)
            self.assertIs(mf.base_field, init_case["base_field"])

    def test_correct_n_fields(self):
        for init_case in self.INIT_CASES:
            mf = TextMultiField(**init_case)
            self.assertEqual(len(mf.fields),
                             len(init_case["feats_fields"]) + 1)

    def test_fields_order_correct(self):
        for init_case in self.INIT_CASES:
            mf = TextMultiField(**init_case)
            fnames = [name for name, _ in init_case["feats_fields"]]
            correct_order = [init_case["base_name"]] + list(sorted(fnames))
            self.assertEqual([name for name, _ in mf.fields], correct_order)

    def test_getitem_0_returns_correct_field(self):
        for init_case in self.INIT_CASES:
            mf = TextMultiField(**init_case)
            self.assertEqual(mf[0][0], init_case["base_name"])
            self.assertIs(mf[0][1], init_case["base_field"])

    def test_getitem_nonzero_returns_correct_field(self):
        for init_case in self.INIT_CASES:
            mf = TextMultiField(**init_case)
            self.assertFalse(True)  # WIP
