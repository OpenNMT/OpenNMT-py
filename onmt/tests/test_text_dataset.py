import unittest
from onmt.inputters.text_dataset import TextMultiField, TextDataReader

import itertools
import os
from copy import deepcopy

from torchtext.data import Field

from onmt.tests.utils_for_tests import product_dict


class TestTextMultiField(unittest.TestCase):
    INIT_CASES = list(product_dict(
        base_name=["base_field", "zbase_field"],
        base_field=[Field],
        feats_fields=[
            [],
            [("a", Field)],
            [("r", Field), ("b", Field)]]))

    PARAMS = list(product_dict(
        include_lengths=[False, True]))

    @classmethod
    def initialize_case(cls, init_case, params):
        # initialize fields at the top of each unit test to prevent
        # any undesired stateful effects
        case = deepcopy(init_case)
        case["base_field"] = case["base_field"](
            include_lengths=params["include_lengths"])
        for i, (n, f_cls) in enumerate(case["feats_fields"]):
            case["feats_fields"][i] = (n, f_cls(sequential=True))
        return case

    def test_process_shape(self):
        dummy_input_bs_1 = [[
                ["this", "is", "for", "the", "unittest"],
                ["NOUN", "VERB", "PREP", "ART", "NOUN"],
                ["", "", "", "", "MODULE"]]]
        dummy_input_bs_5 = [
                [["this", "is", "for", "the", "unittest"],
                 ["NOUN", "VERB", "PREP", "ART", "NOUN"],
                 ["", "", "", "", "MODULE"]],
                [["batch", "2"],
                 ["NOUN", "NUM"],
                 ["", ""]],
                [["batch", "3", "is", "the", "longest", "batch"],
                 ["NOUN", "NUM", "VERB", "ART", "ADJ", "NOUN"],
                 ["", "", "", "", "", ""]],
                [["fourth", "batch"],
                 ["ORD", "NOUN"],
                 ["", ""]],
                [["and", "another", "one"],
                 ["CONJ", "?", "NUM"],
                 ["", "", ""]]]
        for bs, max_len, dummy_input in [
                (1, 5, dummy_input_bs_1), (5, 6, dummy_input_bs_5)]:
            for init_case, params in itertools.product(
                    self.INIT_CASES, self.PARAMS):
                init_case = self.initialize_case(init_case, params)
                mf = TextMultiField(**init_case)
                fields = [init_case["base_field"]] \
                    + [f for _, f in init_case["feats_fields"]]
                nfields = len(fields)
                for i, f in enumerate(fields):
                    all_sents = [b[i] for b in dummy_input]
                    f.build_vocab(all_sents)
                inp_only_desired_fields = [b[:nfields] for b in dummy_input]
                data = mf.process(inp_only_desired_fields)
                if params["include_lengths"]:
                    data, lengths = data
                    self.assertEqual(lengths.shape, (bs,))
                expected_shape = (max_len, bs, nfields)
                self.assertEqual(data.shape, expected_shape)

    def test_preprocess_shape(self):
        for init_case, params in itertools.product(
                self.INIT_CASES, self.PARAMS):
            init_case = self.initialize_case(init_case, params)
            mf = TextMultiField(**init_case)
            sample_str = "dummy input here ."
            proc = mf.preprocess(sample_str)
            self.assertEqual(len(proc), len(init_case["feats_fields"]) + 1)

    def test_base_field(self):
        for init_case, params in itertools.product(
                self.INIT_CASES, self.PARAMS):
            init_case = self.initialize_case(init_case, params)
            mf = TextMultiField(**init_case)
            self.assertIs(mf.base_field, init_case["base_field"])

    def test_correct_n_fields(self):
        for init_case, params in itertools.product(
                self.INIT_CASES, self.PARAMS):
            init_case = self.initialize_case(init_case, params)
            mf = TextMultiField(**init_case)
            self.assertEqual(len(mf.fields),
                             len(init_case["feats_fields"]) + 1)

    def test_fields_order_correct(self):
        for init_case, params in itertools.product(
                self.INIT_CASES, self.PARAMS):
            init_case = self.initialize_case(init_case, params)
            mf = TextMultiField(**init_case)
            fnames = [name for name, _ in init_case["feats_fields"]]
            correct_order = [init_case["base_name"]] + list(sorted(fnames))
            self.assertEqual([name for name, _ in mf.fields], correct_order)

    def test_getitem_0_returns_correct_field(self):
        for init_case, params in itertools.product(
                self.INIT_CASES, self.PARAMS):
            init_case = self.initialize_case(init_case, params)
            mf = TextMultiField(**init_case)
            self.assertEqual(mf[0][0], init_case["base_name"])
            self.assertIs(mf[0][1], init_case["base_field"])

    def test_getitem_nonzero_returns_correct_field(self):
        for init_case, params in itertools.product(
                self.INIT_CASES, self.PARAMS):
            init_case = self.initialize_case(init_case, params)
            mf = TextMultiField(**init_case)
            fnames = [name for name, _ in init_case["feats_fields"]]
            if len(fnames) > 0:
                ordered_names = list(sorted(fnames))
                name2field = dict(init_case["feats_fields"])
                for i, name in enumerate(ordered_names, 1):
                    expected_field = name2field[name]
                    self.assertIs(mf[i][1], expected_field)

    def test_getitem_has_correct_number_of_indexes(self):
        for init_case, params in itertools.product(
                self.INIT_CASES, self.PARAMS):
            init_case = self.initialize_case(init_case, params)
            mf = TextMultiField(**init_case)
            nfields = len(init_case["feats_fields"]) + 1
            with self.assertRaises(IndexError):
                mf[nfields]


class TestTextDataReader(unittest.TestCase):
    def test_read(self):
        strings = [
            "hello world".encode("utf-8"),
            "this's a string with punctuation .".encode("utf-8"),
            "ThIs Is A sTrInG wItH oDD CapitALIZAtion".encode("utf-8")
        ]
        rdr = TextDataReader()
        for i, ex in enumerate(rdr.read(strings, "src")):
            self.assertEqual(ex["src"], strings[i].decode("utf-8"))


class TestTextDataReaderFromFS(unittest.TestCase):
    # this test touches the file system, so it could be considered an
    # integration test
    STRINGS = [
            "hello world\n".encode("utf-8"),
            "this's a string with punctuation . \n".encode("utf-8"),
            "ThIs Is A sTrInG wItH oDD CapitALIZAtion\n".encode("utf-8")
    ]
    FILE_NAME = "test_strings.txt"

    @classmethod
    def setUpClass(cls):
        # write utf-8 bytes
        with open(cls.FILE_NAME, "wb") as f:
            for str_ in cls.STRINGS:
                f.write(str_)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.FILE_NAME)

    def test_read(self):
        rdr = TextDataReader()
        for i, ex in enumerate(rdr.read(self.FILE_NAME, "src")):
            self.assertEqual(ex["src"], self.STRINGS[i].decode("utf-8"))
