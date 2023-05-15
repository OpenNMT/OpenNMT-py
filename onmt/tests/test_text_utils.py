import unittest
from onmt.inputters.text_utils import parse_features


class TestTextUtils(unittest.TestCase):
    def test_parse_features(self):
        input_data = "this is a test"
        text, feats = parse_features(input_data)
        self.assertEqual(text, "this is a test")
        self.assertEqual(feats, None)

        input_data = "this is a test"
        text, feats = parse_features(input_data, 0, "0")
        self.assertEqual(text, "this is a test")
        self.assertEqual(feats, None)

        input_data = "this is a test"
        n_src_feats = 1
        src_feats_defaults = "0"
        text, feats = parse_features(input_data, n_src_feats, src_feats_defaults)
        self.assertEqual(text, "this is a test")
        self.assertEqual(feats, ["0 0 0 0"])

        input_data = "this is a test"
        n_src_feats = 2
        src_feats_defaults = "0￨1"
        text, feats = parse_features(input_data, n_src_feats, src_feats_defaults)
        self.assertEqual(text, "this is a test")
        self.assertEqual(feats, ["0 0 0 0", "1 1 1 1"])

        input_data = "this￨0 is￨0 a￨0 test￨1"
        n_src_feats = 1
        src_feats_defaults = "0￨0"
        text, feats = parse_features(input_data, n_src_feats, src_feats_defaults)
        self.assertEqual(text, "this is a test")
        self.assertEqual(feats, ["0 0 0 1"])

        input_data = "this￨0￨1 is￨0￨2 a￨0￨3 test￨1￨4"
        n_src_feats = 2
        text, feats = parse_features(input_data, n_src_feats, src_feats_defaults)
        self.assertEqual(text, "this is a test")
        self.assertEqual(feats, ["0 0 0 1", "1 2 3 4"])

    def test_invalid_combinations(self):
        # One source feature expected but none given and no default provided
        input_data = "this is a test"
        n_src_feats = 1
        with self.assertRaises(AssertionError):
            parse_features(input_data, n_src_feats)

        # Provided default does not match required features
        input_data = "this is a test"
        n_src_feats = 1
        src_feats_defaults = "0￨0"
        with self.assertRaises(AssertionError):
            parse_features(input_data, n_src_feats)

        # Data not properly annotated.
        # In this case we do not use the default as it might be an error
        input_data = "this￨0 is￨0 a test￨1"
        n_src_feats = 1
        src_feats_defaults = "0"
        with self.assertRaises(AssertionError):
            parse_features(input_data, n_src_feats, src_feats_defaults)
