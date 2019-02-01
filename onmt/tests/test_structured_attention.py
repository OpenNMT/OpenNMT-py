import unittest
from onmt.modules.structured_attention import MatrixTree

import torch


class TestStructuredAttention(unittest.TestCase):
    def test_matrix_tree_marg_pdfs_sum_to_1(self):
        dtree = MatrixTree()
        q = torch.rand(1, 5, 5)
        marg = dtree.forward(q)
        self.assertTrue(
            marg.sum(1).allclose(torch.tensor(1.0)))
