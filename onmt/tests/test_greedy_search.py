import unittest
from onmt.translate.greedy_search import GreedySearch

import torch


class TestGreedySearch(unittest.TestCase):
    BATCH_SZ = 3
    INP_SEQ_LEN = 53
    DEAD_SCORE = -1e20

    BLOCKED_SCORE = -10e20

    def test_doesnt_predict_eos_if_shorter_than_min_len(self):
        # batch 0 will always predict EOS. The other batches will predict
        # non-eos scores.
        for batch_sz in [1, 3]:
            n_words = 100
            _non_eos_idxs = [47]
            valid_score_dist = torch.log_softmax(torch.tensor(
                [6., 5.]), dim=0)
            min_length = 5
            eos_idx = 2
            lengths = torch.randint(0, 30, (batch_sz,))
            samp = GreedySearch(
                0, 1, 2, batch_sz, min_length,
                False, set(), False, 30, 1., 1)
            samp.initialize(torch.zeros(1), lengths)
            all_attns = []
            for i in range(min_length + 4):
                word_probs = torch.full(
                    (batch_sz, n_words), -float('inf'))
                # "best" prediction is eos - that should be blocked
                word_probs[0, eos_idx] = valid_score_dist[0]
                # include at least one prediction OTHER than EOS
                # that is greater than -1e20
                word_probs[0, _non_eos_idxs[0]] = valid_score_dist[1]
                word_probs[1:, _non_eos_idxs[0] + i] = 0

                attns = torch.randn(1, batch_sz, 53)
                all_attns.append(attns)
                samp.advance(word_probs, attns)
                if i < min_length:
                    self.assertTrue(
                        samp.topk_scores[0].allclose(valid_score_dist[1]))
                    self.assertTrue(
                        samp.topk_scores[1:].eq(0).all())
                elif i == min_length:
                    # now batch 0 has ended and no others have
                    self.assertTrue(samp.is_finished[0, :].eq(1).all())
                    self.assertTrue(samp.is_finished[1:, 1:].eq(0).all())
                else:  # i > min_length
                    break

    def test_returns_correct_scores_deterministic(self):
        for batch_sz in [1, 13]:
            for temp in [1., 3.]:
                n_words = 100
                _non_eos_idxs = [47, 51, 13, 88, 99]
                valid_score_dist_1 = torch.log_softmax(torch.tensor(
                    [6., 5., 4., 3., 2., 1.]), dim=0)
                valid_score_dist_2 = torch.log_softmax(torch.tensor(
                    [6., 1.]), dim=0)
                eos_idx = 2
                lengths = torch.randint(0, 30, (batch_sz,))
                samp = GreedySearch(
                    0, 1, 2, batch_sz, 0,
                    False, set(), False, 30, temp, 1)
                samp.initialize(torch.zeros(1), lengths)
                # initial step
                i = 0
                word_probs = torch.full(
                    (batch_sz, n_words), -float('inf'))
                # batch 0 dies on step 0
                word_probs[0, eos_idx] = valid_score_dist_1[0]
                # include at least one prediction OTHER than EOS
                # that is greater than -1e20
                word_probs[0, _non_eos_idxs] = valid_score_dist_1[1:]
                word_probs[1:, _non_eos_idxs[0] + i] = 0

                attns = torch.randn(1, batch_sz, 53)
                samp.advance(word_probs, attns)
                self.assertTrue(samp.is_finished[0].eq(1).all())
                samp.update_finished()
                self.assertEqual(
                    samp.scores[0], [valid_score_dist_1[0] / temp])
                if batch_sz == 1:
                    self.assertTrue(samp.done)
                    continue
                else:
                    self.assertFalse(samp.done)

                # step 2
                i = 1
                word_probs = torch.full(
                    (batch_sz - 1, n_words), -float('inf'))
                # (old) batch 8 dies on step 1
                word_probs[7, eos_idx] = valid_score_dist_2[0]
                word_probs[0:7, _non_eos_idxs[:2]] = valid_score_dist_2
                word_probs[8:, _non_eos_idxs[:2]] = valid_score_dist_2

                attns = torch.randn(1, batch_sz, 53)
                samp.advance(word_probs, attns)

                self.assertTrue(samp.is_finished[7].eq(1).all())
                samp.update_finished()
                self.assertEqual(
                    samp.scores[8], [valid_score_dist_2[0] / temp])

                # step 3
                i = 2
                word_probs = torch.full(
                    (batch_sz - 2, n_words), -float('inf'))
                # everything dies
                word_probs[:, eos_idx] = 0

                attns = torch.randn(1, batch_sz, 53)
                samp.advance(word_probs, attns)

                self.assertTrue(samp.is_finished.eq(1).all())
                samp.update_finished()
                for b in range(batch_sz):
                    if b != 0 and b != 8:
                        self.assertEqual(samp.scores[b], [0])
                self.assertTrue(samp.done)

    def test_returns_correct_scores_non_deterministic(self):
        for batch_sz in [1, 13]:
            for temp in [1., 3.]:
                n_words = 100
                _non_eos_idxs = [47, 51, 13, 88, 99]
                valid_score_dist_1 = torch.log_softmax(torch.tensor(
                    [6., 5., 4., 3., 2., 1.]), dim=0)
                valid_score_dist_2 = torch.log_softmax(torch.tensor(
                    [6., 1.]), dim=0)
                eos_idx = 2
                lengths = torch.randint(0, 30, (batch_sz,))
                samp = GreedySearch(
                    0, 1, 2, batch_sz, 0,
                    False, set(), False, 30, temp, 2)
                samp.initialize(torch.zeros(1), lengths)
                # initial step
                i = 0
                for _ in range(100):
                    word_probs = torch.full(
                        (batch_sz, n_words), -float('inf'))
                    # batch 0 dies on step 0
                    word_probs[0, eos_idx] = valid_score_dist_1[0]
                    # include at least one prediction OTHER than EOS
                    # that is greater than -1e20
                    word_probs[0, _non_eos_idxs] = valid_score_dist_1[1:]
                    word_probs[1:, _non_eos_idxs[0] + i] = 0

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if samp.is_finished[0].eq(1).all():
                        break
                else:
                    self.fail("Batch 0 never ended (very unlikely but maybe "
                              "due to stochasticisty. If so, please increase "
                              "the range of the for-loop.")
                samp.update_finished()
                self.assertEqual(
                    samp.scores[0], [valid_score_dist_1[0] / temp])
                if batch_sz == 1:
                    self.assertTrue(samp.done)
                    continue
                else:
                    self.assertFalse(samp.done)

                # step 2
                i = 1
                for _ in range(100):
                    word_probs = torch.full(
                        (batch_sz - 1, n_words), -float('inf'))
                    # (old) batch 8 dies on step 1
                    word_probs[7, eos_idx] = valid_score_dist_2[0]
                    word_probs[0:7, _non_eos_idxs[:2]] = valid_score_dist_2
                    word_probs[8:, _non_eos_idxs[:2]] = valid_score_dist_2

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if samp.is_finished[7].eq(1).all():
                        break
                else:
                    self.fail("Batch 8 never ended (very unlikely but maybe "
                              "due to stochasticisty. If so, please increase "
                              "the range of the for-loop.")

                samp.update_finished()
                self.assertEqual(
                    samp.scores[8], [valid_score_dist_2[0] / temp])

                # step 3
                i = 2
                for _ in range(250):
                    word_probs = torch.full(
                        (samp.alive_seq.shape[0], n_words), -float('inf'))
                    # everything dies
                    word_probs[:, eos_idx] = 0

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if samp.is_finished.any():
                        samp.update_finished()
                    if samp.is_finished.eq(1).all():
                        break
                else:
                    self.fail("All batches never ended (very unlikely but "
                              "maybe due to stochasticisty. If so, please "
                              "increase the range of the for-loop.")

                for b in range(batch_sz):
                    if b != 0 and b != 8:
                        self.assertEqual(samp.scores[b], [0])
                self.assertTrue(samp.done)
