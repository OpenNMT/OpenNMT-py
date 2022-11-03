"""
Here come the tests for attention types and their compatibility
"""
import unittest
import torch
from torch.autograd import Variable

import onmt


class TestAttention(unittest.TestCase):

    def test_masked_global_attention(self):

        src_len = torch.IntTensor([7, 3, 5, 2])
        # illegal_weights_mask = torch.ByteTensor([
        #     [0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 1, 1, 1, 1],
        #     [0, 0, 0, 0, 0, 1, 1],
        #     [0, 0, 1, 1, 1, 1, 1]])

        batch_size = src_len.size(0)
        dim = 20

        enc_out = Variable(torch.randn(batch_size,
                           src_len.max(), dim))
        enc_final_hs = Variable(torch.randn(batch_size, dim))

        attn = onmt.modules.GlobalAttention(dim)

        _, alignments = attn(enc_final_hs, enc_out,
                             src_len=src_len)
        # TODO: fix for pytorch 0.3
        # illegal_weights = alignments.masked_select(illegal_weights_mask)

        # self.assertEqual(0.0, illegal_weights.data.sum())
