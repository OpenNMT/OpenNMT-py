import unittest
from onmt.translate.beam_search import BeamSearch, GNMTGlobalScorer

from copy import deepcopy

import torch


class GlobalScorerStub(object):
    alpha = 0
    beta = 0

    def __init__(self):
        self.length_penalty = lambda x, alpha: 1.
        self.cov_penalty = lambda cov, beta: torch.zeros(
            (1, cov.shape[-2]), device=cov.device, dtype=torch.float)
        self.has_cov_pen = False
        self.has_len_pen = False

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
        device_init = torch.zeros(1, 1)
        for batch_sz in [1, 3]:
            beam = BeamSearch(
                beam_sz, batch_sz, 0, 1, 2, 2,
                GlobalScorerStub(), 0, 30,
                False, ngram_repeat, set(),
                False, 0.)
            beam.initialize(device_init, torch.randint(0, 30, (batch_sz,)))
            for i in range(ngram_repeat + 4):
                # predict repeat_idx over and over again
                word_probs = torch.full(
                    (batch_sz * beam_sz, n_words), -float('inf'))
                word_probs[0::beam_sz, repeat_idx] = 0

                attns = torch.randn(1, batch_sz * beam_sz, 53)
                beam.advance(word_probs, attns)

                if i < ngram_repeat:
                    # before repeat, scores are either 0 or -inf
                    expected_scores = torch.tensor(
                        [0] + [-float('inf')] * (beam_sz - 1))\
                        .repeat(batch_sz, 1)
                    self.assertTrue(beam.topk_log_probs.equal(expected_scores))
                elif i % ngram_repeat == 0:
                    # on repeat, `repeat_idx` score is BLOCKED_SCORE
                    # (but it's still the best score, thus we have
                    # [BLOCKED_SCORE, -inf, -inf, -inf, -inf]
                    expected_scores = torch.tensor(
                        [0] + [-float('inf')] * (beam_sz - 1))\
                        .repeat(batch_sz, 1)
                    expected_scores[:, 0] = self.BLOCKED_SCORE
                    self.assertTrue(beam.topk_log_probs.equal(expected_scores))
                else:
                    # repetitions keeps maximizing score
                    # index 0 has been blocked, so repeating=>+0.0 score
                    # other indexes are -inf so repeating=>BLOCKED_SCORE
                    # which is higher
                    expected_scores = torch.tensor(
                        [0] + [-float('inf')] * (beam_sz - 1))\
                        .repeat(batch_sz, 1)
                    expected_scores[:, :] = self.BLOCKED_SCORE
                    expected_scores = torch.tensor(
                        self.BLOCKED_SCORE).repeat(batch_sz, beam_sz)

    def test_advance_with_some_repeats_gets_blocked(self):
        # beam 0 and beam >=2 will repeat (beam >= 2 repeat dummy scores)
        beam_sz = 5
        n_words = 100
        repeat_idx = 47
        ngram_repeat = 3
        no_repeat_score = -2.3
        repeat_score = -0.1
        device_init = torch.zeros(1, 1)
        for batch_sz in [1, 3]:
            beam = BeamSearch(
                beam_sz, batch_sz, 0, 1, 2, 2,
                GlobalScorerStub(), 0, 30,
                False, ngram_repeat, set(),
                False, 0.)
            beam.initialize(device_init, torch.randint(0, 30, (batch_sz,)))
            for i in range(ngram_repeat + 4):
                # non-interesting beams are going to get dummy values
                word_probs = torch.full(
                    (batch_sz * beam_sz, n_words), -float('inf'))
                if i == 0:
                    # on initial round, only predicted scores for beam 0
                    # matter. Make two predictions. Top one will be repeated
                    # in beam zero, second one will live on in beam 1.
                    word_probs[0::beam_sz, repeat_idx] = repeat_score
                    word_probs[0::beam_sz, repeat_idx +
                               i + 1] = no_repeat_score
                else:
                    # predict the same thing in beam 0
                    word_probs[0::beam_sz, repeat_idx] = 0
                    # continue pushing around what beam 1 predicts
                    word_probs[1::beam_sz, repeat_idx + i + 1] = 0
                attns = torch.randn(1, batch_sz * beam_sz, 53)
                beam.advance(word_probs, attns)
                if i < ngram_repeat:
                    self.assertFalse(
                        beam.topk_log_probs[0::beam_sz].eq(
                            self.BLOCKED_SCORE).any())
                    self.assertFalse(
                        beam.topk_log_probs[1::beam_sz].eq(
                            self.BLOCKED_SCORE).any())
                elif i == ngram_repeat:
                    # now beam 0 dies (along with the others), beam 1 -> beam 0
                    self.assertFalse(
                        beam.topk_log_probs[:, 0].eq(
                            self.BLOCKED_SCORE).any())

                    expected = torch.full([batch_sz, beam_sz], float("-inf"))
                    expected[:, 0] = no_repeat_score
                    expected[:, 1] = self.BLOCKED_SCORE
                    self.assertTrue(
                        beam.topk_log_probs[:, :].equal(expected))
                else:
                    # now beam 0 dies (along with the others), beam 1 -> beam 0
                    self.assertFalse(
                        beam.topk_log_probs[:, 0].eq(
                            self.BLOCKED_SCORE).any())

                    expected = torch.full([batch_sz, beam_sz], float("-inf"))
                    expected[:, 0] = no_repeat_score
                    expected[:, 1:] = self.BLOCKED_SCORE
                    self.assertTrue(
                        beam.topk_log_probs.equal(expected))

    def test_repeating_excluded_index_does_not_die(self):
        # beam 0 and beam >= 2 will repeat (beam 2 repeats excluded idx)
        beam_sz = 5
        n_words = 100
        repeat_idx = 47  # will be repeated and should be blocked
        repeat_idx_ignored = 7  # will be repeated and should not be blocked
        ngram_repeat = 3
        device_init = torch.zeros(1, 1)
        for batch_sz in [1, 3]:
            beam = BeamSearch(
                beam_sz, batch_sz, 0, 1, 2, 2,
                GlobalScorerStub(), 0, 30,
                False, ngram_repeat, {repeat_idx_ignored},
                False, 0.)
            beam.initialize(device_init, torch.randint(0, 30, (batch_sz,)))
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
                attns = torch.randn(1, batch_sz * beam_sz, 53)
                beam.advance(word_probs, attns)
                if i < ngram_repeat:
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

                    self.assertTrue(beam.topk_log_probs[:, 2].eq(
                        self.BLOCKED_SCORE).all())

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
            lengths = torch.randint(0, 30, (batch_sz,))
            beam = BeamSearch(beam_sz, batch_sz, 0, 1, 2, 2,
                              GlobalScorerStub(),
                              min_length, 30, False, 0, set(),
                              False, 0.)
            device_init = torch.zeros(1, 1)
            beam.initialize(device_init, lengths)
            all_attns = []
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
                        beam_idx = min(beam_sz - 1, k)
                        word_probs[beam_idx::beam_sz, j] = score

                attns = torch.randn(1, batch_sz * beam_sz, 53)
                all_attns.append(attns)
                beam.advance(word_probs, attns)
                if i < min_length:
                    expected_score_dist = \
                        (i + 1) * valid_score_dist[1:].unsqueeze(0)
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
        beam = BeamSearch(
            beam_sz, batch_sz, 0, 1, 2, 2,
            GlobalScorerStub(),
            min_length, 30, False, 0, set(),
            False, 0.)
        device_init = torch.zeros(1, 1)
        beam.initialize(device_init, torch.randint(0, 30, (batch_sz,)))
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
                    beam_idx = min(beam_sz - 1, k)
                    word_probs[beam_idx::beam_sz, j] = score
            else:
                word_probs[0::beam_sz, eos_idx] = valid_score_dist[0]
                word_probs[1::beam_sz, eos_idx] = valid_score_dist[0]
                # provide beam_sz other good predictions in other beams
                for k, (j, score) in enumerate(
                        zip(_non_eos_idxs, valid_score_dist[1:])):
                    beam_idx = min(beam_sz - 1, k)
                    word_probs[beam_idx::beam_sz, j] = score

            attns = torch.randn(1, batch_sz * beam_sz, 53)
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

    def test_beam_returns_attn_with_correct_length(self):
        beam_sz = 5
        batch_sz = 3
        n_words = 100
        _non_eos_idxs = [47, 51, 13, 88, 99]
        valid_score_dist = torch.log_softmax(torch.tensor(
            [6., 5., 4., 3., 2., 1.]), dim=0)
        min_length = 5
        eos_idx = 2
        inp_lens = torch.randint(1, 30, (batch_sz,))
        beam = BeamSearch(
            beam_sz, batch_sz, 0, 1, 2, 2,
            GlobalScorerStub(),
            min_length, 30, True, 0, set(),
            False, 0.)
        device_init = torch.zeros(1, 1)
        _, _, inp_lens, _ = beam.initialize(device_init, inp_lens)
        # inp_lens is tiled in initialize, reassign to make attn match
        for i in range(min_length + 2):
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
                    beam_idx = min(beam_sz - 1, k)
                    word_probs[beam_idx::beam_sz, j] = score
            else:
                word_probs[0::beam_sz, eos_idx] = valid_score_dist[0]
                word_probs[1::beam_sz, eos_idx] = valid_score_dist[0]
                # provide beam_sz other good predictions in other beams
                for k, (j, score) in enumerate(
                        zip(_non_eos_idxs, valid_score_dist[1:])):
                    beam_idx = min(beam_sz - 1, k)
                    word_probs[beam_idx::beam_sz, j] = score

            attns = torch.randn(1, batch_sz * beam_sz, 53)
            beam.advance(word_probs, attns)
            if i < min_length:
                self.assertFalse(beam.done)
                # no top beams are finished yet
                for b in range(batch_sz):
                    self.assertEqual(beam.attention[b], [])
            elif i == min_length:
                # beam 1 dies on min_length
                self.assertTrue(beam.is_finished[:, 1].all())
                beam.update_finished()
                self.assertFalse(beam.done)
                # no top beams are finished yet
                for b in range(batch_sz):
                    self.assertEqual(beam.attention[b], [])
            else:  # i > min_length
                # beam 0 dies on the step after beam 1 dies
                self.assertTrue(beam.is_finished[:, 0].all())
                beam.update_finished()
                self.assertTrue(beam.done)
                # top beam is finished now so there are attentions
                for b in range(batch_sz):
                    # two beams are finished in each batch
                    self.assertEqual(len(beam.attention[b]), 2)
                    for k in range(2):
                        # second dim is cut down to the non-padded src length
                        self.assertEqual(beam.attention[b][k].shape[-1],
                                         inp_lens[b])
                    # first dim is equal to the time of death
                    # (beam 0 died at current step - adjust for SOS)
                    self.assertEqual(beam.attention[b][0].shape[0], i + 1)
                    # (beam 1 died at last step - adjust for SOS)
                    self.assertEqual(beam.attention[b][1].shape[0], i)
                # behavior gets weird when beam is already done so just stop
                break


class TestBeamSearchAgainstReferenceCase(unittest.TestCase):
    # this is just test_beam.TestBeamAgainstReferenceCase repeated
    # in each batch.
    BEAM_SZ = 5
    EOS_IDX = 2  # don't change this - all the scores would need updated
    N_WORDS = 8  # also don't change for same reason
    N_BEST = 3
    DEAD_SCORE = -1e20
    BATCH_SZ = 3
    INP_SEQ_LEN = 53

    def random_attn(self):
        return torch.randn(1, self.BATCH_SZ * self.BEAM_SZ, self.INP_SEQ_LEN)

    def init_step(self, beam, expected_len_pen):
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

    def first_step(self, beam, expected_beam_scores, expected_len_pen):
        # no EOS's yet
        assert beam.is_finished.sum() == 0
        scores_1 = torch.log_softmax(torch.tensor(
            [[0, 0, 0, .3, 0, .51, .2, 0],
             [0, 0, 1.5, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, .49, .48, 0, 0],
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
        self.assertTrue(beam.topk_scores.allclose(
            expected_beam_scores / expected_len_pen))
        self.assertTrue(beam.topk_ids.equal(expected_preds_1))
        self.assertTrue(beam.current_backptr.equal(expected_bptr_1))
        self.assertEqual(beam.is_finished.sum(), self.BATCH_SZ)
        self.assertTrue(beam.is_finished[:, 2].all())  # beam 2 finished
        beam.update_finished()
        self.assertFalse(beam.top_beam_finished.any())
        self.assertFalse(beam.done)
        return expected_beam_scores

    def second_step(self, beam, expected_beam_scores, expected_len_pen):
        # assumes beam 2 finished on last step
        scores_2 = torch.log_softmax(torch.tensor(
            [[0, 0, 0, .3, 0, .51, .2, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 5000, .48, 0, 0],  # beam 2 shouldn't continue
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
        self.assertTrue(beam.topk_scores.allclose(
            expected_beam_scores / expected_len_pen))
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

    def third_step(self, beam, expected_beam_scores, expected_len_pen):
        # assumes beam 0 finished on last step
        scores_3 = torch.log_softmax(torch.tensor(
            [[0, 0, 5000, 0, 5000, .51, .2, 0],  # beam 0 shouldn't cont
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 5000, 0, 0],
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
        self.assertTrue(beam.topk_log_probs.allclose(
            expected_beam_scores))
        self.assertTrue(beam.topk_scores.allclose(
            expected_beam_scores / expected_len_pen))
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
            GlobalScorerStub(),
            0, 30, False, 0, set(),
            False, 0.)
        device_init = torch.zeros(1, 1)
        beam.initialize(device_init, torch.randint(0, 30, (self.BATCH_SZ,)))
        expected_beam_scores = self.init_step(beam, 1)
        expected_beam_scores = self.first_step(beam, expected_beam_scores, 1)
        expected_beam_scores = self.second_step(beam, expected_beam_scores, 1)
        self.third_step(beam, expected_beam_scores, 1)


class TestBeamWithLengthPenalty(TestBeamSearchAgainstReferenceCase):
    # this could be considered an integration test because it tests
    # interactions between the GNMT scorer and the beam

    def test_beam_advance_against_known_reference(self):
        scorer = GNMTGlobalScorer(0.7, 0., "avg", "none")
        beam = BeamSearch(
            self.BEAM_SZ, self.BATCH_SZ, 0, 1, 2, self.N_BEST,
            scorer,
            0, 30, False, 0, set(),
            False, 0.)
        device_init = torch.zeros(1, 1)
        beam.initialize(device_init, torch.randint(0, 30, (self.BATCH_SZ,)))
        expected_beam_scores = self.init_step(beam, 1.)
        expected_beam_scores = self.first_step(beam, expected_beam_scores, 3)
        expected_beam_scores = self.second_step(beam, expected_beam_scores, 4)
        self.third_step(beam, expected_beam_scores, 5)
