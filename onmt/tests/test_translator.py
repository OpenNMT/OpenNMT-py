import unittest
from onmt.translate import GeneratorLM
import torch


class TestGeneratorLM(unittest.TestCase):
    def test_split_src_to_prevent_padding_target_prefix_is_none_when_equal_size(  # noqa: E501
        self,
    ):
        src = torch.randint(0, 10, (6, 5, 1))
        src_lengths = 5 * torch.ones(5, dtype=torch.int)
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
        src = torch.randint(0, 10, (6, default_length, 1))
        src_lengths = default_length * torch.ones(6, dtype=torch.int)
        new_length = 4
        src_lengths[1] = new_length
        (
            src,
            src_lengths,
            target_prefix,
        ) = GeneratorLM.split_src_to_prevent_padding(src, src_lengths)
        self.assertTupleEqual(src.shape, (6, new_length, 1))
        self.assertTupleEqual(target_prefix.shape, (6, 1, 1))
        self.assertTrue(
            src_lengths.equal(new_length * torch.ones(6, dtype=torch.int))
        )
