import unittest
from onmt.translate import GeneratorLM
import torch


class TestGeneratorLM(unittest.TestCase):
    def test_split_src_to_prevent_padding_target_prefix_is_none_when_equal_size(  # noqa: E501
        self,
    ):
        src = torch.randint(0, 10, (5, 6))
        src_lengths = 5 * torch.ones(5)
        (
            src,
            src_lengths,
            target_prefix,
        ) = GeneratorLM.split_src_to_prevent_padding(src, src_lengths)
        self.assertIsNone(target_prefix)

    def test_split_src_to_prevent_padding_target_prefix_is_ok_when_different_size(  # noqa: E501
        self,
    ):
        default_length = 5
        src = torch.randint(0, 10, (default_length, 6))
        src_lengths = default_length * torch.ones(6, dtype=torch.int)
        new_length = 4
        src_lengths[1] = new_length
        (
            src,
            src_lengths,
            target_prefix,
        ) = GeneratorLM.split_src_to_prevent_padding(src, src_lengths)
        self.assertTupleEqual(src.shape, (new_length, 6))
        self.assertTupleEqual(target_prefix.shape, (1, 6))
        self.assertTrue(
            src_lengths.equal(new_length * torch.ones(6, dtype=torch.int))
        )
