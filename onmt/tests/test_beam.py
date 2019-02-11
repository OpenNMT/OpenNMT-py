import unittest
from onmt.translate.beam import Beam, GNMTGlobalScorer

import torch


class GlobalScorerStub(object):
    def update_global_state(self, beam):
        pass

    def score(self, beam, scores):
        return scores


class TestBeam(unittest.TestCase):
    BLOCKED_SCORE = -10e20

    def test_advance_with_all_repeats_gets_blocked(self):
        # all beams repeat (beam >= 1 repeat dummy scores)
        beam_sz = 5
        n_words = 100
        repeat_idx = 47
        ngram_repeat = 3
        beam = Beam(beam_sz, 0, 1, 2, n_best=2,
                    exclusion_tokens=set(),
                    global_scorer=GlobalScorerStub(),
                    block_ngram_repeat=ngram_repeat)
        for i in range(ngram_repeat + 4):
            # predict repeat_idx over and over again
            word_probs = torch.full((beam_sz, n_words), -float('inf'))
            word_probs[0, repeat_idx] = 0
            attns = torch.randn(beam_sz)
            beam.advance(word_probs, attns)
            if i <= ngram_repeat:
                self.assertTrue(
                    beam.scores.equal(
                        torch.tensor(
                            [0] + [-float('inf')] * (beam_sz - 1))))
            else:
                self.assertTrue(
                    beam.scores.equal(torch.tensor(
                        [self.BLOCKED_SCORE] * beam_sz)))

    def test_advance_with_some_repeats_gets_blocked(self):
        # beam 0 and beam >=2 will repeat (beam >= 2 repeat dummy scores)
        beam_sz = 5
        n_words = 100
        repeat_idx = 47
        ngram_repeat = 3
        beam = Beam(beam_sz, 0, 1, 2, n_best=2,
                    exclusion_tokens=set(),
                    global_scorer=GlobalScorerStub(),
                    block_ngram_repeat=ngram_repeat)
        for i in range(ngram_repeat + 4):
            # non-interesting beams are going to get dummy values
            word_probs = torch.full((beam_sz, n_words), -float('inf'))
            if i == 0:
                # on initial round, only predicted scores for beam 0
                # matter. Make two predictions. Top one will be repeated
                # in beam zero, second one will live on in beam 1.
                word_probs[0, repeat_idx] = -0.1
                word_probs[0, repeat_idx + i + 1] = -2.3
            else:
                # predict the same thing in beam 0
                word_probs[0, repeat_idx] = 0
                # continue pushing around what beam 1 predicts
                word_probs[1, repeat_idx + i + 1] = 0
            attns = torch.randn(beam_sz)
            beam.advance(word_probs, attns)
            if i <= ngram_repeat:
                self.assertFalse(beam.scores[0].eq(self.BLOCKED_SCORE))
                self.assertFalse(beam.scores[1].eq(self.BLOCKED_SCORE))
            else:
                # now beam 0 dies (along with the others), beam 1 -> beam 0
                self.assertFalse(beam.scores[0].eq(self.BLOCKED_SCORE))
                self.assertTrue(
                    beam.scores[1:].equal(torch.tensor(
                        [self.BLOCKED_SCORE] * (beam_sz - 1))))

    def test_repeating_excluded_index_does_not_die(self):
        # beam 0 and beam >= 2 will repeat (beam 2 repeats excluded idx)
        beam_sz = 5
        n_words = 100
        repeat_idx = 47  # will be repeated and should be blocked
        repeat_idx_ignored = 7  # will be repeated and should not be blocked
        ngram_repeat = 3
        beam = Beam(beam_sz, 0, 1, 2, n_best=2,
                    exclusion_tokens=set([repeat_idx_ignored]),
                    global_scorer=GlobalScorerStub(),
                    block_ngram_repeat=ngram_repeat)
        for i in range(ngram_repeat + 4):
            # non-interesting beams are going to get dummy values
            word_probs = torch.full((beam_sz, n_words), -float('inf'))
            if i == 0:
                word_probs[0, repeat_idx] = -0.1
                word_probs[0, repeat_idx + i + 1] = -2.3
                word_probs[0, repeat_idx_ignored] = -5.0
            else:
                # predict the same thing in beam 0
                word_probs[0, repeat_idx] = 0
                # continue pushing around what beam 1 predicts
                word_probs[1, repeat_idx + i + 1] = 0
                # predict the allowed-repeat again in beam 2
                word_probs[2, repeat_idx_ignored] = 0
            attns = torch.randn(beam_sz)
            beam.advance(word_probs, attns)
            if i <= ngram_repeat:
                self.assertFalse(beam.scores[0].eq(self.BLOCKED_SCORE))
                self.assertFalse(beam.scores[1].eq(self.BLOCKED_SCORE))
                self.assertFalse(beam.scores[2].eq(self.BLOCKED_SCORE))
            else:
                # now beam 0 dies, beam 1 -> beam 0, beam 2 -> beam 1
                # and the rest die
                self.assertFalse(beam.scores[0].eq(self.BLOCKED_SCORE))
                # since all preds after i=0 are 0, we can check
                # that the beam is the correct idx by checking that
                # the curr score is the initial score
                self.assertTrue(beam.scores[0].eq(-2.3))
                self.assertFalse(beam.scores[1].eq(self.BLOCKED_SCORE))
                self.assertTrue(beam.scores[1].eq(-5.0))
                self.assertTrue(
                    beam.scores[2:].equal(torch.tensor(
                        [self.BLOCKED_SCORE] * (beam_sz - 2))))

    def test_doesnt_predict_eos_if_shorter_than_min_len(self):
        # beam 0 will always predict EOS. The other beams will predict
        # non-eos scores.
        # this is also a test that when block_ngram_repeat=0,
        # repeating is acceptable
        beam_sz = 5
        n_words = 100
        _non_eos_idxs = [47, 51, 13, 88, 99]
        valid_score_dist = torch.log_softmax(torch.tensor(
            [6., 5., 4., 3., 2., 1.]), dim=0)
        min_length = 5
        eos_idx = 2
        beam = Beam(beam_sz, 0, 1, eos_idx, n_best=2,
                    exclusion_tokens=set(),
                    min_length=min_length,
                    global_scorer=GlobalScorerStub(),
                    block_ngram_repeat=0)
        for i in range(min_length + 4):
            # non-interesting beams are going to get dummy values
            word_probs = torch.full((beam_sz, n_words), -float('inf'))
            if i == 0:
                # "best" prediction is eos - that should be blocked
                word_probs[0, eos_idx] = valid_score_dist[0]
                # include at least beam_sz predictions OTHER than EOS
                # that are greater than -1e20
                for j, score in zip(_non_eos_idxs, valid_score_dist[1:]):
                    word_probs[0, j] = score
            else:
                # predict eos in beam 0
                word_probs[0, eos_idx] = valid_score_dist[0]
                # provide beam_sz other good predictions
                for k, (j, score) in enumerate(
                        zip(_non_eos_idxs, valid_score_dist[1:])):
                    beam_idx = min(beam_sz-1, k)
                    word_probs[beam_idx, j] = score

            attns = torch.randn(beam_sz)
            beam.advance(word_probs, attns)
            if i < min_length:
                expected_score_dist = (i+1) * valid_score_dist[1:]
                self.assertTrue(beam.scores.allclose(expected_score_dist))
            elif i == min_length:
                # now the top beam has ended and no others have
                # first beam finished had length beam.min_length
                self.assertEqual(beam.finished[0][1], beam.min_length + 1)
                # first beam finished was 0
                self.assertEqual(beam.finished[0][2], 0)
            else:  # i > min_length
                # not of interest, but want to make sure it keeps running
                # since only beam 0 terminates and n_best = 2
                pass

    def test_beam_is_done_when_n_best_beams_eos_using_min_length(self):
        # this is also a test that when block_ngram_repeat=0,
        # repeating is acceptable
        beam_sz = 5
        n_words = 100
        _non_eos_idxs = [47, 51, 13, 88, 99]
        valid_score_dist = torch.log_softmax(torch.tensor(
            [6., 5., 4., 3., 2., 1.]), dim=0)
        min_length = 5
        eos_idx = 2
        beam = Beam(beam_sz, 0, 1, eos_idx, n_best=2,
                    exclusion_tokens=set(),
                    min_length=min_length,
                    global_scorer=GlobalScorerStub(),
                    block_ngram_repeat=0)
        for i in range(min_length + 4):
            # non-interesting beams are going to get dummy values
            word_probs = torch.full((beam_sz, n_words), -float('inf'))
            if i == 0:
                # "best" prediction is eos - that should be blocked
                word_probs[0, eos_idx] = valid_score_dist[0]
                # include at least beam_sz predictions OTHER than EOS
                # that are greater than -1e20
                for j, score in zip(_non_eos_idxs, valid_score_dist[1:]):
                    word_probs[0, j] = score
            elif i <= min_length:
                # predict eos in beam 1
                word_probs[1, eos_idx] = valid_score_dist[0]
                # provide beam_sz other good predictions in other beams
                for k, (j, score) in enumerate(
                        zip(_non_eos_idxs, valid_score_dist[1:])):
                    beam_idx = min(beam_sz-1, k)
                    word_probs[beam_idx, j] = score
            else:
                word_probs[0, eos_idx] = valid_score_dist[0]
                word_probs[1, eos_idx] = valid_score_dist[0]
                # provide beam_sz other good predictions in other beams
                for k, (j, score) in enumerate(
                        zip(_non_eos_idxs, valid_score_dist[1:])):
                    beam_idx = min(beam_sz-1, k)
                    word_probs[beam_idx, j] = score

            attns = torch.randn(beam_sz)
            beam.advance(word_probs, attns)
            if i < min_length:
                self.assertFalse(beam.done)
            elif i == min_length:
                # beam 1 dies on min_length
                self.assertEqual(beam.finished[0][1], beam.min_length + 1)
                self.assertEqual(beam.finished[0][2], 1)
                self.assertFalse(beam.done)
            else:  # i > min_length
                # beam 0 dies on the step after beam 1 dies
                self.assertEqual(beam.finished[1][1], beam.min_length + 2)
                self.assertEqual(beam.finished[1][2], 0)
                self.assertTrue(beam.done)


class TestBeamAgainstReferenceCase(unittest.TestCase):
    BEAM_SZ = 5
    EOS_IDX = 2  # don't change this - all the scores would need updated
    N_WORDS = 8  # also don't change for same reason
    N_BEST = 3
    DEAD_SCORE = -1e20
    INP_SEQ_LEN = 53

    def init_step(self, beam):
        # init_preds: [4, 3, 5, 6, 7] - no EOS's
        init_scores = torch.log_softmax(torch.tensor(
            [[0, 0, 0, 4, 5, 3, 2, 1]], dtype=torch.float), dim=1)
        expected_beam_scores, expected_preds_0 = init_scores.topk(self.BEAM_SZ)
        beam.advance(init_scores, torch.randn(self.BEAM_SZ, self.INP_SEQ_LEN))
        self.assertTrue(beam.scores.allclose(expected_beam_scores))
        self.assertTrue(beam.next_ys[-1].equal(expected_preds_0[0]))
        self.assertFalse(beam.eos_top)
        self.assertFalse(beam.done)
        return expected_beam_scores

    def first_step(self, beam, expected_beam_scores, expected_len_pen):
        # no EOS's yet
        assert len(beam.finished) == 0
        scores_1 = torch.log_softmax(torch.tensor(
            [[0, 0,  0, .3,   0, .51, .2, 0],
             [0, 0, 1.5,  0,   0,   0,  0, 0],
             [0, 0,  0,  0, .49, .48,  0, 0],
             [0, 0, 0, .2, .2, .2, .2, .2],
             [0, 0, 0, .2, .2, .2, .2, .2]]
        ), dim=1)

        beam.advance(scores_1, torch.randn(self.BEAM_SZ, self.INP_SEQ_LEN))

        new_scores = scores_1 + expected_beam_scores.t()
        expected_beam_scores, unreduced_preds = new_scores.view(-1).topk(
            self.BEAM_SZ, 0, True, True)
        expected_bptr_1 = unreduced_preds / self.N_WORDS
        # [5, 3, 2, 6, 0], so beam 2 predicts EOS!
        expected_preds_1 = unreduced_preds - expected_bptr_1 * self.N_WORDS

        self.assertTrue(beam.scores.allclose(expected_beam_scores))
        self.assertTrue(beam.next_ys[-1].equal(expected_preds_1))
        self.assertTrue(beam.prev_ks[-1].equal(expected_bptr_1))
        self.assertEqual(len(beam.finished), 1)
        self.assertEqual(beam.finished[0][2], 2)  # beam 2 finished
        self.assertEqual(beam.finished[0][1], 2)  # finished on second step
        self.assertEqual(beam.finished[0][0],  # finished with correct score
                         expected_beam_scores[2] / expected_len_pen)
        self.assertFalse(beam.eos_top)
        self.assertFalse(beam.done)
        return expected_beam_scores

    def second_step(self, beam, expected_beam_scores, expected_len_pen):
        # assumes beam 2 finished on last step
        scores_2 = torch.log_softmax(torch.tensor(
            [[0, 0,  0, .3,   0, .51, .2, 0],
             [0, 0, 0,  0,   0,   0,  0, 0],
             [0, 0,  0,  0, 5000, .48,  0, 0],  # beam 2 shouldn't continue
             [0, 0, 50, .2, .2, .2, .2, .2],  # beam 3 -> beam 0 should die
             [0, 0, 0, .2, .2, .2, .2, .2]]
        ), dim=1)

        beam.advance(scores_2, torch.randn(self.BEAM_SZ, self.INP_SEQ_LEN))

        new_scores = scores_2 + expected_beam_scores.unsqueeze(1)
        new_scores[2] = self.DEAD_SCORE  # ended beam 2 shouldn't continue
        expected_beam_scores, unreduced_preds = new_scores.view(-1).topk(
            self.BEAM_SZ, 0, True, True)
        expected_bptr_2 = unreduced_preds / self.N_WORDS
        # [2, 5, 3, 6, 0], so beam 0 predicts EOS!
        expected_preds_2 = unreduced_preds - expected_bptr_2 * self.N_WORDS
        # [-2.4879, -3.8910, -4.1010, -4.2010, -4.4010]
        self.assertTrue(beam.scores.allclose(expected_beam_scores))
        self.assertTrue(beam.next_ys[-1].equal(expected_preds_2))
        self.assertTrue(beam.prev_ks[-1].equal(expected_bptr_2))
        self.assertEqual(len(beam.finished), 2)
        # new beam 0 finished
        self.assertEqual(beam.finished[1][2], 0)
        # new beam 0 is old beam 3
        self.assertEqual(expected_bptr_2[0], 3)
        self.assertEqual(beam.finished[1][1], 3)  # finished on third step
        self.assertEqual(beam.finished[1][0],  # finished with correct score
                         expected_beam_scores[0] / expected_len_pen)
        self.assertTrue(beam.eos_top)
        self.assertFalse(beam.done)
        return expected_beam_scores

    def third_step(self, beam, expected_beam_scores, expected_len_pen):
        # assumes beam 0 finished on last step
        scores_3 = torch.log_softmax(torch.tensor(
            [[0, 0,  5000, 0,   5000, .51, .2, 0],  # beam 0 shouldn't cont
             [0, 0, 0,  0,   0,   0,  0, 0],
             [0, 0,  0,  0, 0, 5000,  0, 0],
             [0, 0, 0, .2, .2, .2, .2, .2],
             [0, 0, 50, 0, .2, .2, .2, .2]]  # beam 4 -> beam 1 should die
        ), dim=1)

        beam.advance(scores_3, torch.randn(self.BEAM_SZ, self.INP_SEQ_LEN))

        new_scores = scores_3 + expected_beam_scores.unsqueeze(1)
        new_scores[0] = self.DEAD_SCORE  # ended beam 2 shouldn't continue
        expected_beam_scores, unreduced_preds = new_scores.view(-1).topk(
            self.BEAM_SZ, 0, True, True)
        expected_bptr_3 = unreduced_preds / self.N_WORDS
        # [5, 2, 6, 1, 0], so beam 1 predicts EOS!
        expected_preds_3 = unreduced_preds - expected_bptr_3 * self.N_WORDS
        self.assertTrue(beam.scores.allclose(expected_beam_scores))
        self.assertTrue(beam.next_ys[-1].equal(expected_preds_3))
        self.assertTrue(beam.prev_ks[-1].equal(expected_bptr_3))
        self.assertEqual(len(beam.finished), 3)
        # new beam 1 finished
        self.assertEqual(beam.finished[2][2], 1)
        # new beam 1 is old beam 4
        self.assertEqual(expected_bptr_3[1], 4)
        self.assertEqual(beam.finished[2][1], 4)  # finished on fourth step
        self.assertEqual(beam.finished[2][0],  # finished with correct score
                         expected_beam_scores[1] / expected_len_pen)
        self.assertTrue(beam.eos_top)
        self.assertTrue(beam.done)
        return expected_beam_scores

    def test_beam_advance_against_known_reference(self):
        beam = Beam(self.BEAM_SZ, 0, 1, self.EOS_IDX, n_best=self.N_BEST,
                    exclusion_tokens=set(),
                    min_length=0,
                    global_scorer=GlobalScorerStub(),
                    block_ngram_repeat=0)

        expected_beam_scores = self.init_step(beam)
        expected_beam_scores = self.first_step(beam, expected_beam_scores, 1)
        expected_beam_scores = self.second_step(beam, expected_beam_scores, 1)
        self.third_step(beam, expected_beam_scores, 1)


class TestBeamWithLengthPenalty(TestBeamAgainstReferenceCase):
    # this could be considered an integration test because it tests
    # interactions between the GNMT scorer and the beam

    def test_beam_advance_against_known_reference(self):
        scorer = GNMTGlobalScorer(0.7, 0., "avg", "none")
        beam = Beam(self.BEAM_SZ, 0, 1, self.EOS_IDX, n_best=self.N_BEST,
                    exclusion_tokens=set(),
                    min_length=0,
                    global_scorer=scorer,
                    block_ngram_repeat=0)
        expected_beam_scores = self.init_step(beam)
        expected_beam_scores = self.first_step(beam, expected_beam_scores, 3)
        expected_beam_scores = self.second_step(beam, expected_beam_scores, 4)
        self.third_step(beam, expected_beam_scores, 5)
