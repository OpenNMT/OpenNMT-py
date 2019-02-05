import unittest
from onmt.translate.beam_search import BeamSearch

from copy import deepcopy

import torch


class GlobalScorerStub(object):
    alpha = 0

    def update_global_state(self, beam):
        pass

    def score(self, beam, scores):
        return scores


class TestBeamSearch(unittest.TestCase):
    BLOCKED_SCORE = -10e20

    def test_advance_with_all_repeats_gets_blocked(self):
        # all beams repeat (beam >= 1 repeat dummy scores)
        beam_sz = 5
        n_words = 100
        repeat_idx = 47
        ngram_repeat = 3
        for batch_sz in [1, 3]:
            beam = BeamSearch(
                beam_sz, batch_sz, 0, 1, 2, 2,
                torch.device("cpu"), GlobalScorerStub(), 0, 30,
                False, ngram_repeat, set(),
                torch.randint(0, 30, (batch_sz,)))
            for i in range(ngram_repeat + 4):
                # predict repeat_idx over and over again
                word_probs = torch.full(
                    (batch_sz * beam_sz, n_words), -float('inf'))
                word_probs[0::beam_sz, repeat_idx] = 0
                attns = torch.randn(beam_sz)
                beam.advance(word_probs, attns)
                if i <= ngram_repeat:
                    expected_scores = torch.tensor(
                                [0] + [-float('inf')] * (beam_sz - 1))\
                            .repeat(batch_sz, 1)
                    self.assertTrue(beam.topk_log_probs.equal(expected_scores))
                else:
                    self.assertTrue(
                        beam.topk_log_probs.equal(
                            torch.tensor(self.BLOCKED_SCORE)
                            .repeat(batch_sz, beam_sz)))

    def test_advance_with_some_repeats_gets_blocked(self):
        # beam 0 and beam >=2 will repeat (beam >= 2 repeat dummy scores)
        beam_sz = 5
        n_words = 100
        repeat_idx = 47
        ngram_repeat = 3
        for batch_sz in [1, 3]:
            beam = BeamSearch(
                beam_sz, batch_sz, 0, 1, 2, 2,
                torch.device("cpu"), GlobalScorerStub(), 0, 30,
                False, ngram_repeat, set(),
                torch.randint(0, 30, (batch_sz,)))
            for i in range(ngram_repeat + 4):
                # non-interesting beams are going to get dummy values
                word_probs = torch.full(
                    (batch_sz * beam_sz, n_words), -float('inf'))
                if i == 0:
                    # on initial round, only predicted scores for beam 0
                    # matter. Make two predictions. Top one will be repeated
                    # in beam zero, second one will live on in beam 1.
                    word_probs[0::beam_sz, repeat_idx] = -0.1
                    word_probs[0::beam_sz, repeat_idx + i + 1] = -2.3
                else:
                    # predict the same thing in beam 0
                    word_probs[0::beam_sz, repeat_idx] = 0
                    # continue pushing around what beam 1 predicts
                    word_probs[1::beam_sz, repeat_idx + i + 1] = 0
                attns = torch.randn(beam_sz)
                beam.advance(word_probs, attns)
                if i <= ngram_repeat:
                    self.assertFalse(
                        beam.topk_log_probs[0::beam_sz].eq(
                            self.BLOCKED_SCORE).any())
                    self.assertFalse(
                        beam.topk_log_probs[1::beam_sz].eq(
                            self.BLOCKED_SCORE).any())
                else:
                    # now beam 0 dies (along with the others), beam 1 -> beam 0
                    self.assertFalse(
                        beam.topk_log_probs[:, 0].eq(
                            self.BLOCKED_SCORE).any())
                    self.assertTrue(
                        beam.topk_log_probs[:, 1:].equal(
                            torch.tensor(self.BLOCKED_SCORE)
                            .repeat(batch_sz, beam_sz-1)))

    def test_repeating_excluded_index_does_not_die(self):
        # beam 0 and beam >= 2 will repeat (beam 2 repeats excluded idx)
        beam_sz = 5
        n_words = 100
        repeat_idx = 47  # will be repeated and should be blocked
        repeat_idx_ignored = 7  # will be repeated and should not be blocked
        ngram_repeat = 3
        for batch_sz in [1, 3]:
            beam = BeamSearch(
                beam_sz, batch_sz, 0, 1, 2, 2,
                torch.device("cpu"), GlobalScorerStub(), 0, 30,
                False, ngram_repeat, {repeat_idx_ignored},
                torch.randint(0, 30, (batch_sz,)))
            for i in range(ngram_repeat + 4):
                # non-interesting beams are going to get dummy values
                word_probs = torch.full(
                    (batch_sz * beam_sz, n_words), -float('inf'))
                if i == 0:
                    word_probs[0::beam_sz, repeat_idx] = -0.1
                    word_probs[0::beam_sz, repeat_idx + i + 1] = -2.3
                    word_probs[0::beam_sz, repeat_idx_ignored] = -5.0
                else:
                    # predict the same thing in beam 0
                    word_probs[0::beam_sz, repeat_idx] = 0
                    # continue pushing around what beam 1 predicts
                    word_probs[1::beam_sz, repeat_idx + i + 1] = 0
                    # predict the allowed-repeat again in beam 2
                    word_probs[2::beam_sz, repeat_idx_ignored] = 0
                attns = torch.randn(beam_sz)
                beam.advance(word_probs, attns)
                if i <= ngram_repeat:
                    self.assertFalse(beam.topk_log_probs[:, 0].eq(
                        self.BLOCKED_SCORE).any())
                    self.assertFalse(beam.topk_log_probs[:, 1].eq(
                        self.BLOCKED_SCORE).any())
                    self.assertFalse(beam.topk_log_probs[:, 2].eq(
                        self.BLOCKED_SCORE).any())
                else:
                    # now beam 0 dies, beam 1 -> beam 0, beam 2 -> beam 1
                    # and the rest die
                    self.assertFalse(beam.topk_log_probs[:, 0].eq(
                        self.BLOCKED_SCORE).any())
                    # since all preds after i=0 are 0, we can check
                    # that the beam is the correct idx by checking that
                    # the curr score is the initial score
                    self.assertTrue(beam.topk_log_probs[:, 0].eq(-2.3).all())
                    self.assertFalse(beam.topk_log_probs[:, 1].eq(
                        self.BLOCKED_SCORE).all())
                    self.assertTrue(beam.topk_log_probs[:, 1].eq(-5.0).all())
                    self.assertTrue(
                        beam.topk_log_probs[:, 2:].equal(
                            torch.tensor(self.BLOCKED_SCORE)
                            .repeat(batch_sz, beam_sz - 2)))

    def test_doesnt_predict_eos_if_shorter_than_min_len(self):
        # beam 0 will always predict EOS. The other beams will predict
        # non-eos scores.
        for batch_sz in [1, 3]:
            beam_sz = 5
            n_words = 100
            _non_eos_idxs = [47, 51, 13, 88, 99]
            valid_score_dist = torch.log_softmax(torch.tensor(
                [6., 5., 4., 3., 2., 1.]), dim=0)
            min_length = 5
            eos_idx = 2
            # beam includes start token in cur_len count.
            # Add one to its min_length to compensate
            beam = BeamSearch(beam_sz, batch_sz, 0, 1, 2, 2,
                              torch.device("cpu"), GlobalScorerStub(),
                              min_length + 1, 30, False, 0, set(),
                              torch.randint(0, 30, (batch_sz,)))
            for i in range(min_length + 4):
                # non-interesting beams are going to get dummy values
                word_probs = torch.full(
                    (batch_sz * beam_sz, n_words), -float('inf'))
                if i == 0:
                    # "best" prediction is eos - that should be blocked
                    word_probs[0::beam_sz, eos_idx] = valid_score_dist[0]
                    # include at least beam_sz predictions OTHER than EOS
                    # that are greater than -1e20
                    for j, score in zip(_non_eos_idxs, valid_score_dist[1:]):
                        word_probs[0::beam_sz, j] = score
                else:
                    # predict eos in beam 0
                    word_probs[0::beam_sz, eos_idx] = valid_score_dist[0]
                    # provide beam_sz other good predictions
                    for k, (j, score) in enumerate(
                            zip(_non_eos_idxs, valid_score_dist[1:])):
                        beam_idx = min(beam_sz-1, k)
                        word_probs[beam_idx::beam_sz, j] = score

                attns = torch.randn(beam_sz)
                beam.advance(word_probs, attns)
                if i < min_length:
                    expected_score_dist = \
                        (i+1) * valid_score_dist[1:].unsqueeze(0)
                    self.assertTrue(
                        beam.topk_log_probs.allclose(
                            expected_score_dist))
                elif i == min_length:
                    # now the top beam has ended and no others have
                    self.assertTrue(beam.is_finished[:, 0].eq(1).all())
                    self.assertTrue(beam.is_finished[:, 1:].eq(0).all())
                else:  # i > min_length
                    # not of interest, but want to make sure it keeps running
                    # since only beam 0 terminates and n_best = 2
                    pass

    def test_beam_is_done_when_n_best_beams_eos_using_min_length(self):
        # this is also a test that when block_ngram_repeat=0,
        # repeating is acceptable
        beam_sz = 5
        batch_sz = 3
        n_words = 100
        _non_eos_idxs = [47, 51, 13, 88, 99]
        valid_score_dist = torch.log_softmax(torch.tensor(
            [6., 5., 4., 3., 2., 1.]), dim=0)
        min_length = 5
        eos_idx = 2
        # beam includes start token in cur_len count.
        # Add one to its min_length to compensate
        beam = BeamSearch(
            beam_sz, batch_sz, 0, 1, 2, 2,
            torch.device("cpu"), GlobalScorerStub(),
            min_length + 1, 30, False, 0, set(),
            torch.randint(0, 30, (batch_sz,)))
        for i in range(min_length + 4):
            # non-interesting beams are going to get dummy values
            word_probs = torch.full(
                (batch_sz * beam_sz, n_words), -float('inf'))
            if i == 0:
                # "best" prediction is eos - that should be blocked
                word_probs[0::beam_sz, eos_idx] = valid_score_dist[0]
                # include at least beam_sz predictions OTHER than EOS
                # that are greater than -1e20
                for j, score in zip(_non_eos_idxs, valid_score_dist[1:]):
                    word_probs[0::beam_sz, j] = score
            elif i <= min_length:
                # predict eos in beam 1
                word_probs[1::beam_sz, eos_idx] = valid_score_dist[0]
                # provide beam_sz other good predictions in other beams
                for k, (j, score) in enumerate(
                        zip(_non_eos_idxs, valid_score_dist[1:])):
                    beam_idx = min(beam_sz-1, k)
                    word_probs[beam_idx::beam_sz, j] = score
            else:
                word_probs[0::beam_sz, eos_idx] = valid_score_dist[0]
                word_probs[1::beam_sz, eos_idx] = valid_score_dist[0]
                # provide beam_sz other good predictions in other beams
                for k, (j, score) in enumerate(
                        zip(_non_eos_idxs, valid_score_dist[1:])):
                    beam_idx = min(beam_sz-1, k)
                    word_probs[beam_idx::beam_sz, j] = score

            attns = torch.randn(beam_sz)
            beam.advance(word_probs, attns)
            if i < min_length:
                self.assertFalse(beam.done)
            elif i == min_length:
                # beam 1 dies on min_length
                self.assertTrue(beam.is_finished[:, 1].all())
                beam.update_finished()
                self.assertFalse(beam.done)
            else:  # i > min_length
                # beam 0 dies on the step after beam 1 dies
                self.assertTrue(beam.is_finished[:, 0].all())
                beam.update_finished()
                self.assertTrue(beam.done)


class TestBeamSearchAgainstReferenceCase(unittest.TestCase):
    # this is just test_beam.TestBeamAgainstReferenceCase repeated
    # in each batch.
    BEAM_SZ = 5
    EOS_IDX = 2  # don't change this - all the scores would need updated
    N_WORDS = 8  # also don't change for same reason
    N_BEST = 3
    DEAD_SCORE = -1e20
    BATCH_SZ = 3

    def random_attn(self):
        return torch.randn(self.BATCH_SZ, self.BEAM_SZ)

    def init_step(self, beam):
        # init_preds: [4, 3, 5, 6, 7] - no EOS's
        init_scores = torch.log_softmax(torch.tensor(
            [[0, 0, 0, 4, 5, 3, 2, 1]], dtype=torch.float), dim=1)
        init_scores = deepcopy(init_scores.repeat(
            self.BATCH_SZ * self.BEAM_SZ, 1))
        new_scores = init_scores + beam.topk_log_probs.view(-1).unsqueeze(1)
        expected_beam_scores, expected_preds_0 = new_scores \
            .view(self.BATCH_SZ, self.BEAM_SZ * self.N_WORDS) \
            .topk(self.BEAM_SZ, dim=-1)
        beam.advance(deepcopy(init_scores), self.random_attn())
        self.assertTrue(beam.topk_log_probs.allclose(expected_beam_scores))
        self.assertTrue(beam.topk_ids.equal(expected_preds_0))
        self.assertFalse(beam.is_finished.any())
        self.assertFalse(beam.done)
        return expected_beam_scores

    def first_step(self, beam, expected_beam_scores):
        # no EOS's yet
        assert beam.is_finished.sum() == 0
        scores_1 = torch.log_softmax(torch.tensor(
            [[0, 0,  0, .3,   0, .51, .2, 0],
             [0, 0, 1.5,  0,   0,   0,  0, 0],
             [0, 0,  0,  0, .49, .48,  0, 0],
             [0, 0, 0, .2, .2, .2, .2, .2],
             [0, 0, 0, .2, .2, .2, .2, .2]]
        ), dim=1)
        scores_1 = scores_1.repeat(self.BATCH_SZ, 1)

        beam.advance(deepcopy(scores_1), self.random_attn())

        new_scores = scores_1 + expected_beam_scores.view(-1).unsqueeze(1)
        expected_beam_scores, unreduced_preds = new_scores\
            .view(self.BATCH_SZ, self.BEAM_SZ * self.N_WORDS)\
            .topk(self.BEAM_SZ, -1)
        expected_bptr_1 = unreduced_preds / self.N_WORDS
        # [5, 3, 2, 6, 0], so beam 2 predicts EOS!
        expected_preds_1 = unreduced_preds - expected_bptr_1 * self.N_WORDS

        self.assertTrue(beam.topk_log_probs.allclose(expected_beam_scores))
        self.assertTrue(beam.topk_ids.equal(expected_preds_1))
        self.assertTrue(beam.current_backptr.equal(expected_bptr_1))
        self.assertEqual(beam.is_finished.sum(), self.BATCH_SZ)
        self.assertTrue(beam.is_finished[:, 2].all())  # beam 2 finished
        beam.update_finished()
        self.assertFalse(beam.top_beam_finished.any())
        self.assertFalse(beam.done)
        return expected_beam_scores

    def second_step(self, beam, expected_beam_scores):
        # assumes beam 2 finished on last step
        scores_2 = torch.log_softmax(torch.tensor(
            [[0, 0,  0, .3,   0, .51, .2, 0],
             [0, 0, 0,  0,   0,   0,  0, 0],
             [0, 0,  0,  0, 5000, .48,  0, 0],  # beam 2 shouldn't continue
             [0, 0, 50, .2, .2, .2, .2, .2],  # beam 3 -> beam 0 should die
             [0, 0, 0, .2, .2, .2, .2, .2]]
        ), dim=1)
        scores_2 = scores_2.repeat(self.BATCH_SZ, 1)

        beam.advance(deepcopy(scores_2), self.random_attn())

        # ended beam 2 shouldn't continue
        expected_beam_scores[:, 2::self.BEAM_SZ] = self.DEAD_SCORE
        new_scores = scores_2 + expected_beam_scores.view(-1).unsqueeze(1)
        expected_beam_scores, unreduced_preds = new_scores\
            .view(self.BATCH_SZ, self.BEAM_SZ * self.N_WORDS)\
            .topk(self.BEAM_SZ, -1)
        expected_bptr_2 = unreduced_preds / self.N_WORDS
        # [2, 5, 3, 6, 0] repeat self.BATCH_SZ, so beam 0 predicts EOS!
        expected_preds_2 = unreduced_preds - expected_bptr_2 * self.N_WORDS
        # [-2.4879, -3.8910, -4.1010, -4.2010, -4.4010] repeat self.BATCH_SZ
        self.assertTrue(beam.topk_log_probs.allclose(expected_beam_scores))
        self.assertTrue(beam.topk_ids.equal(expected_preds_2))
        self.assertTrue(beam.current_backptr.equal(expected_bptr_2))
        # another beam is finished in all batches
        self.assertEqual(beam.is_finished.sum(), self.BATCH_SZ)
        # new beam 0 finished
        self.assertTrue(beam.is_finished[:, 0].all())
        # new beam 0 is old beam 3
        self.assertTrue(expected_bptr_2[:, 0].eq(3).all())
        beam.update_finished()
        self.assertTrue(beam.top_beam_finished.all())
        self.assertFalse(beam.done)
        return expected_beam_scores

    def third_step(self, beam, expected_beam_scores):
        # assumes beam 0 finished on last step
        scores_3 = torch.log_softmax(torch.tensor(
            [[0, 0,  5000, 0,   5000, .51, .2, 0],  # beam 0 shouldn't cont
             [0, 0, 0,  0,   0,   0,  0, 0],
             [0, 0,  0,  0, 0, 5000,  0, 0],
             [0, 0, 0, .2, .2, .2, .2, .2],
             [0, 0, 50, 0, .2, .2, .2, .2]]  # beam 4 -> beam 1 should die
        ), dim=1)
        scores_3 = scores_3.repeat(self.BATCH_SZ, 1)

        beam.advance(deepcopy(scores_3), self.random_attn())

        expected_beam_scores[:, 0::self.BEAM_SZ] = self.DEAD_SCORE
        new_scores = scores_3 + expected_beam_scores.view(-1).unsqueeze(1)
        expected_beam_scores, unreduced_preds = new_scores\
            .view(self.BATCH_SZ, self.BEAM_SZ * self.N_WORDS)\
            .topk(self.BEAM_SZ, -1)
        expected_bptr_3 = unreduced_preds / self.N_WORDS
        # [5, 2, 6, 1, 0] repeat self.BATCH_SZ, so beam 1 predicts EOS!
        expected_preds_3 = unreduced_preds - expected_bptr_3 * self.N_WORDS
        self.assertTrue(beam.topk_log_probs.allclose(expected_beam_scores))
        self.assertTrue(beam.topk_ids.equal(expected_preds_3))
        self.assertTrue(beam.current_backptr.equal(expected_bptr_3))
        self.assertEqual(beam.is_finished.sum(), self.BATCH_SZ)
        # new beam 1 finished
        self.assertTrue(beam.is_finished[:, 1].all())
        # new beam 1 is old beam 4
        self.assertTrue(expected_bptr_3[:, 1].eq(4).all())
        beam.update_finished()
        self.assertTrue(beam.top_beam_finished.all())
        self.assertTrue(beam.done)
        return expected_beam_scores

    def test_beam_advance_against_known_reference(self):
        beam = BeamSearch(
            self.BEAM_SZ, self.BATCH_SZ, 0, 1, 2, self.N_BEST,
            torch.device("cpu"), GlobalScorerStub(),
            0, 30, False, 0, set(),
            torch.randint(0, 30, (self.BATCH_SZ,)))

        expected_beam_scores = self.init_step(beam)
        expected_beam_scores = self.first_step(beam, expected_beam_scores)
        expected_beam_scores = self.second_step(beam, expected_beam_scores)
        self.third_step(beam, expected_beam_scores)
