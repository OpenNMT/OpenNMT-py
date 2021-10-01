import unittest

from onmt.transforms.bart import word_start_finder
from onmt.utils.alignment import subword_map_by_joiner, subword_map_by_spacer
from onmt.constants import SubwordMarker


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
        out = subword_map_by_joiner(data_in, marker=SubwordMarker.JOINER)
        self.assertEqual(out, true_out)

    def test_subword_group_joiner_with_case_markup(self):
        data_in = ['｟mrk_case_modifier_C｠', 'however', '￭,', 'according', 'to', 'the', 'logs', '￭,', '｟mrk_begin_case_region_U｠', 'she', 'is', 'hard', '￭-￭', 'working', '｟mrk_end_case_region_U｠', '￭.']  # noqa: E501
        true_out = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 6, 7, 7, 7, 7, 7]
        out = subword_map_by_joiner(data_in, marker=SubwordMarker.JOINER)
        self.assertEqual(out, true_out)

    def test_subword_group_joiner_with_case_markup_advanced(self):
        data_in = ['｟mrk_case_modifier_C｠', 'dummy', 'text', '｟mrk_case_modifier_C｠', '1￭', 'h￭', 'k', '｟mrk_begin_case_region_U｠', 'th￭', '｟mrk_end_case_region_U｠', 'n', 'more', 'dummy', 'text']  # noqa: E501
        true_out = [0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6]
        out = subword_map_by_joiner(
            data_in,
            marker=SubwordMarker.JOINER)
        self.assertEqual(out, true_out)

    def test_subword_group_joiner_prior_tokenization(self):
        data_in = ['｟mrk_case_modifier_C｠', 'how￭', 'ever', '￭,', 'according', 'to', 'the', 'logs', '￭,', '｟mrk_begin_case_region_U｠', 'she', 'is', 'hard', '￭-￭', 'working', '｟mrk_end_case_region_U｠', '￭.']  # noqa: E501
        original_data_in = ['However', '￭,', 'according', 'to', 'the', 'logs', '￭,', 'SHE', 'IS', 'HARD-WORKING', '￭.']  # noqa: E501
        true_out = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 9, 9, 9, 10]  # noqa: E501
        out = subword_map_by_joiner(
            data_in,
            marker=SubwordMarker.JOINER,
            original_subwords=original_data_in)
        self.assertEqual(out, true_out)

    def test_subword_group_joiner_prior_tokenization_harder(self):
        data_in = ['｟mrk_case_modifier_C｠', 'how￭', 'ever', '￭,', 'according', 'to', 'the', 'logs', '￭,', '｟mrk_begin_case_region_U｠', 'she', 'is', 'hard', '￭-￭', 'working', '｟mrk_end_case_region_U｠', '￭.']  # noqa: E501
        original_data_in = ['｟mrk_case_modifier_C｠', 'how￭', 'ever', '￭,', 'according', 'to', 'the', 'logs', '￭,', '｟mrk_begin_case_region_U｠', 'she', 'is', 'hard', '￭-￭', 'working', '｟mrk_end_case_region_U｠', '￭.']  # noqa: E501
        true_out = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # noqa: E501
        out = subword_map_by_joiner(
            data_in,
            marker=SubwordMarker.JOINER,
            original_subwords=original_data_in)
        self.assertEqual(out, true_out)

    def test_subword_group_joiner_with_new_joiner(self):
        data_in = ['｟mrk_case_modifier_C｠', 'however', '￭', ',', 'according', 'to', 'the', 'logs', '￭', ',', '｟mrk_begin_case_region_U｠', 'she', 'is', 'hard', '￭', '-', '￭', 'working', '｟mrk_end_case_region_U｠', '￭', '.']  # noqa: E501
        true_out = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7]  # noqa: E501
        out = subword_map_by_joiner(data_in, marker=SubwordMarker.JOINER)
        self.assertEqual(out, true_out)

    def test_subword_group_naive(self):
        data_in = ['however', ',', 'according', 'to', 'the', 'logs', ',', 'she', 'is', 'hard', '-', 'working', '.']  # noqa: E501
        true_out = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        out = subword_map_by_joiner(data_in, marker=SubwordMarker.JOINER)
        self.assertEqual(out, true_out)

    def test_subword_group_spacer(self):
        data_in = ['however', ',', '▁according', '▁to', '▁the', '▁logs', ',', '▁she', '▁is', '▁hard', '-', 'working', '.']  # noqa: E501
        true_out = [0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 7, 7, 7]
        out = subword_map_by_spacer(data_in, marker=SubwordMarker.SPACER)
        self.assertEqual(out, true_out)
        # no dummy prefix
        no_dummy = ['however', ',', '▁according', '▁to', '▁the', '▁logs', ',', '▁she', '▁is', '▁hard', '-', 'working', '.']  # noqa: E501
        no_dummy_out = subword_map_by_spacer(
            no_dummy, marker=SubwordMarker.SPACER)
        self.assertEqual(no_dummy_out, true_out)

    def test_subword_group_spacer_with_case_markup(self):
        data_in = ['｟mrk_case_modifier_C｠', '▁however', ',', '▁according', '▁to', '▁the', '▁logs', ',', '▁｟mrk_begin_case_region_U｠', '▁she', '▁is', '▁hard', '-', 'working', '.', '▁｟mrk_end_case_region_U｠']  # noqa: E501
        true_out = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 6, 7, 7, 7, 7, 7]
        out = subword_map_by_spacer(
            data_in, marker=SubwordMarker.SPACER)
        self.assertEqual(out, true_out)

    def test_subword_group_spacer_with_spacer_new(self):
        data_in = ['｟mrk_case_modifier_C｠', '▁', 'however', ',', '▁', 'according', '▁', 'to', '▁', 'the', '▁', 'logs', ',', '▁', '｟mrk_begin_case_region_U｠', '▁', 'she', '▁', 'is', '▁', 'hard', '-', 'working', '.', '▁', '｟mrk_end_case_region_U｠']  # noqa: E501
        true_out = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7]  # noqa: E501
        out = subword_map_by_spacer(
            data_in, marker=SubwordMarker.SPACER)
        self.assertEqual(out, true_out)


if __name__ == '__main__':
    unittest.main()
