import unittest

from onmt.transforms.bart import word_start_finder
from onmt.utils.alignment import subword_map_by_joiner, subword_map_by_spacer


class TestWordStartFinder(unittest.TestCase):

    def test_word_start_naive(self):
        word_start_finder_fn = word_start_finder(ignore_subword=True)
        data_in = ['however', ',', 'according', 'to', 'the', 'logs', ',', 'she', 'is', 'hard', '-', 'working', '.']  # noqa: E501
        true_out = [True, True, True, True, True, True, True, True, True, True, True, True, True]  # noqa: E501
        out = word_start_finder_fn(data_in)
        self.assertEqual(out, true_out)

    def test_word_start_joiner(self):
        word_start_finder_fn = word_start_finder(is_joiner=True)
        data_in = ['however', '￭,', 'according', 'to', 'the', 'logs', '￭,', 'she', 'is', 'hard', '￭-￭', 'working', '￭.']  # noqa: E501
        true_out = [True, False, True, True, True, True, False, True, True, True, False, False, False]  # noqa: E501
        out = word_start_finder_fn(data_in)
        self.assertEqual(out, true_out)

    def test_word_start_spacer(self):
        word_start_finder_fn = word_start_finder()
        data_in = ['▁however', ',', '▁according', '▁to', '▁the', '▁logs', ',', '▁she', '▁is', '▁hard', '-', 'working', '.']  # noqa: E501
        true_out = [True, False, True, True, True, True, False, True, True, True, False, False, False]  # noqa: E501
        out = word_start_finder_fn(data_in)
        self.assertEqual(out, true_out)
        # no dummy prefix
        no_dummy = ['however', ',', '▁according', '▁to', '▁the', '▁logs', ',', '▁she', '▁is', '▁hard', '-', 'working', '.']  # noqa: E501
        no_dummy_out = word_start_finder_fn(no_dummy)
        self.assertEqual(no_dummy_out, true_out)


class TestSubwordGroup(unittest.TestCase):

    def test_subword_group_joiner(self):
        data_in = ['however', '￭,', 'according', 'to', 'the', 'logs', '￭,', 'she', 'is', 'hard', '￭-￭', 'working', '￭.']  # noqa: E501
        true_out = [0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 7, 7, 7]
        out = subword_map_by_joiner(data_in)
        self.assertEqual(out, true_out)

    def test_subword_group_spacer(self):
        data_in = ['however', ',', '▁according', '▁to', '▁the', '▁logs', ',', '▁she', '▁is', '▁hard', '-', 'working', '.']  # noqa: E501
        true_out = [0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 7, 7, 7]
        out = subword_map_by_spacer(data_in)
        self.assertEqual(out, true_out)
        # no dummy prefix
        no_dummy = ['however', ',', '▁according', '▁to', '▁the', '▁logs', ',', '▁she', '▁is', '▁hard', '-', 'working', '.']  # noqa: E501
        no_dummy_out = subword_map_by_spacer(no_dummy)
        self.assertEqual(no_dummy_out, true_out)


if __name__ == '__main__':
    unittest.main()
