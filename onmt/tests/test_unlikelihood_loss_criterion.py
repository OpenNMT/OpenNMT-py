import unittest
from onmt.utils.loss import UnlikelihoodTokenLoss
import torch
import math


class TestUnlikelihoodLossCriterion(unittest.TestCase):
    def test_compute_previous_context_tokens(self):
        criterion = UnlikelihoodTokenLoss(1, 7)
        target = torch.tensor([[2, 3, 4, 3, 5], [1, 1, 5, 6, 7]]).permute(1, 0)
        previous_context_tokens = criterion.compute_previous_context_tokens(
            target
        )

        self.assertEqual(
            previous_context_tokens.permute(1, 0, 2).tolist(),
            torch.tensor(
                [
                    [
                        [7, 7, 7, 7, 7],
                        [2, 7, 7, 7, 7],
                        [2, 3, 7, 7, 7],
                        [2, 7, 4, 7, 7],
                        [2, 3, 4, 3, 7],
                    ],
                    [
                        [7, 7, 7, 7, 7],
                        [7, 7, 7, 7, 7],
                        [1, 1, 7, 7, 7],
                        [1, 1, 5, 7, 7],
                        [7, 7, 7, 7, 7],
                    ],
                ]
            ).tolist(),
        )

    def test_loss_perfect_pred_should_be_zero(self):
        criterion = UnlikelihoodTokenLoss(1, 7)
        n_prob = -10e6
        target = torch.tensor([[2, 3, 4, 3, 5], [1, 1, 5, 6, 7]]).permute(1, 0)
        perfect_probs = [
            [[n_prob if i != t else 1 for i in range(8)] for t in ex_target]
            for ex_target in target
        ]

        # check padded seq is removed
        perfect_probs[-1][-1][-1] = n_prob
        perfect_probs[-1][-1][1] = 0.1

        output = torch.tensor(perfect_probs).view(-1, 8)

        unlikelihood_loss = criterion.compute_unlikelihood_loss(output, target)

        self.assertEqual(unlikelihood_loss.sum().item(), 0)

    def test_loss_value(self):
        criterion = UnlikelihoodTokenLoss(1, 7)
        n_prob = -10e6
        target = torch.tensor([[2, 3, 4, 3, 5], [1, 1, 5, 6, 7]]).permute(1, 0)
        perfect_probs = [
            [[n_prob if i != t else 1 for i in range(8)] for t in ex_target]
            for ex_target in target
        ]

        # check padded seq is removed
        perfect_probs[-1][-1][-1] = n_prob
        perfect_probs[-1][-1][1] = 0.1

        # set prob at 0.5 on 1 after softmax
        perfect_probs[2][-1][1] = 1

        output = torch.tensor(perfect_probs).view(-1, 8)

        unlikelihood_loss = criterion.compute_unlikelihood_loss(output, target)

        self.assertAlmostEqual(
            unlikelihood_loss.view(5, 2, 8)[2, -1, 1].item(), -math.log(0.5)
        )
