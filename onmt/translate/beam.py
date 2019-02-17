from __future__ import division
import torch
from onmt.translate import penalties

import warnings


class Beam(object):
    """Class for managing the internals of the beam search process.

    Takes care of beams, back pointers, and scores.

    Args:
        size (int): Number of beams to use.
        pad (int): Magic integer in output vocab.
        bos (int): Magic integer in output vocab.
        eos (int): Magic integer in output vocab.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        cuda (bool): use gpu
        global_scorer (onmt.translate.GNMTGlobalScorer): Scorer instance.
        min_length (int): Shortest acceptable generation, not counting
            begin-of-sentence or end-of-sentence.
        stepwise_penalty (bool): Apply coverage penalty at every step.
        block_ngram_repeat (int): Block beams where
            ``block_ngram_repeat``-grams repeat.
        exclusion_tokens (set[int]): If a gram contains any of these
            token indices, it may repeat.
    """

    def __init__(self, size, pad, bos, eos,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 exclusion_tokens=set()):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                        .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

        # Apply Penalty at every step
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens

    @property
    def current_predictions(self):
        return self.next_ys[-1]

    @property
    def current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    def advance(self, word_probs, attn_out):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Args:
            word_probs (FloatTensor): probs of advancing from the last step
                ``(K, words)``
            attn_out (FloatTensor): attention at the last step

        Returns:
            bool: True if beam search is complete.
        """

        num_words = word_probs.size(1)
        if self.stepwise_penalty:
            self.global_scorer.update_score(self, attn_out)
        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len <= self.min_length:
            # assumes there are len(word_probs) predictions OTHER
            # than EOS that are greater than -1e20
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + self.scores.unsqueeze(1)
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20

            # Block ngram repeats
            if self.block_ngram_repeat > 0:
                le = len(self.next_ys)
                for j in range(self.next_ys[-1].size(0)):
                    hyp, _ = self.get_hyp(le - 1, j)
                    ngrams = set()
                    fail = False
                    gram = []
                    for i in range(le - 1):
                        # Last n tokens, n = block_ngram_repeat
                        gram = (gram +
                                [hyp[i].item()])[-self.block_ngram_repeat:]
                        # Skip the blocking if it is in the exclusion list
                        if set(gram) & self.exclusion_tokens:
                            continue
                        if tuple(gram) in ngrams:
                            fail = True
                        ngrams.add(tuple(gram))
                    if fail:
                        beam_scores[j] = -10e20
        else:
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                            True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))
        self.attn.append(attn_out.index_select(0, prev_k))
        self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True

    @property
    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """Walk back to construct the full hypothesis."""
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])


class GNMTGlobalScorer(object):
    """NMT re-ranking.

    Args:
       alpha (float): Length parameter.
       beta (float):  Coverage parameter.
       length_penalty (str): Length penalty strategy.
       coverage_penalty (str): Coverage penalty strategy.

    Attributes:
        alpha (float): See above.
        beta (float): See above.
        length_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        coverage_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        has_cov_pen (bool): See :class:`penalties.PenaltyBuilder`.
        has_len_pen (bool): See :class:`penalties.PenaltyBuilder`.
    """

    @classmethod
    def from_opt(cls, opt):
        return cls(
            opt.alpha,
            opt.beta,
            opt.length_penalty,
            opt.coverage_penalty)

    def __init__(self, alpha, beta, length_penalty, coverage_penalty):
        self._validate(alpha, beta, length_penalty, coverage_penalty)
        self.alpha = alpha
        self.beta = beta
        penalty_builder = penalties.PenaltyBuilder(coverage_penalty,
                                                   length_penalty)
        self.has_cov_pen = penalty_builder.has_cov_pen
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty

        self.has_len_pen = penalty_builder.has_len_pen
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty

    @classmethod
    def _validate(cls, alpha, beta, length_penalty, coverage_penalty):
        # these warnings indicate that either the alpha/beta
        # forces a penalty to be a no-op, or a penalty is a no-op but
        # the alpha/beta would suggest otherwise.
        if length_penalty is None or length_penalty == "none":
            if alpha != 0:
                warnings.warn("Non-default `alpha` with no length penalty. "
                              "`alpha` has no effect.")
        else:
            # using some length penalty
            if length_penalty == "wu" and alpha == 0.:
                warnings.warn("Using length penalty Wu with alpha==0 "
                              "is equivalent to using length penalty none.")
        if coverage_penalty is None or coverage_penalty == "none":
            if beta != 0:
                warnings.warn("Non-default `beta` with no coverage penalty. "
                              "`beta` has no effect.")
        else:
            # using some coverage penalty
            if beta == 0.:
                warnings.warn("Non-default coverage penalty with beta==0 "
                              "is equivalent to using coverage penalty none.")

    def score(self, beam, logprobs):
        """Rescore a prediction based on penalty functions."""
        len_pen = self.length_penalty(len(beam.next_ys), self.alpha)
        normalized_probs = logprobs / len_pen
        if not beam.stepwise_penalty:
            penalty = self.cov_penalty(beam.global_state["coverage"],
                                       self.beta)
            normalized_probs -= penalty

        return normalized_probs

    def update_score(self, beam, attn):
        """Update scores of a Beam that is not finished."""
        if "prev_penalty" in beam.global_state.keys():
            beam.scores.add_(beam.global_state["prev_penalty"])
            penalty = self.cov_penalty(beam.global_state["coverage"] + attn,
                                       self.beta)
            beam.scores.sub_(penalty)

    def update_global_state(self, beam):
        """Keeps the coverage vector as sum of attentions."""
        if len(beam.prev_ks) == 1:
            beam.global_state["prev_penalty"] = beam.scores.clone().fill_(0.0)
            beam.global_state["coverage"] = beam.attn[-1]
            self.cov_total = beam.attn[-1].sum(1)
        else:
            self.cov_total += torch.min(beam.attn[-1],
                                        beam.global_state['coverage']).sum(1)
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])

            prev_penalty = self.cov_penalty(beam.global_state["coverage"],
                                            self.beta)
            beam.global_state["prev_penalty"] = prev_penalty
