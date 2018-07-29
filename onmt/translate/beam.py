from __future__ import division
import torch
from onmt.translate import penalties
from onmt.utils.misc import aeq


class Beam(object):
    """
    Class for managing the internals of the beam search process.

    Takes care of beams, back pointers, and scores.

    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """

    def __init__(self, size, pad, bos, eos,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 exclusion_tokens=frozenset()):

        device = 'cuda' if cuda else 'cpu'  # not sure if necessary

        # The score for each translation on the beam.
        # TODO: get rid of one of these attributes
        self.scores = torch.zeros(size, device=device)

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), pad, device=device).long()]
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
    def width(self):
        return len(self.next_ys[-1])

    @property
    def current_state(self):
        return self.next_ys[-1]

    @property
    def backpointers(self):
        return self.prev_ks[-1]

    def advance(self, probs, attn_out):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:
        * `probs`- probs of advancing from the last step
            (beam width x vocab size)
        * `attn_out`- attention at the last step
        """
        probs_beam_width, vocab_size = probs.size()
        aeq(probs_beam_width, self.width)

        if self.stepwise_penalty:
            self.update_score(attn_out)
        # force the output to be longer than self.min_length
        if len(self.next_ys) < self.min_length:
            probs[:, self._eos] = -1e20

        # compute the beam
        beam_scores = self._successors(probs) if self.prev_ks else probs[0]

        # prune the beam
        self.scores, best_scores_id = beam_scores.topk(self.width, 0)

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / vocab_size
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * vocab_size)
        self.attn.append(attn_out.index_select(0, prev_k))
        self.update_global_state()

        finished_sents = self.current_state == self._eos
        if finished_sents.any():
            indices = torch.arange(0, self.width, dtype=torch.long)
            seq_len = len(self.next_ys) - 1

            global_scores = self.compute_global_score()
            finished_globals = global_scores.masked_select(finished_sents)
            ix = indices.masked_select(finished_sents)
            self.finished.extend((s, seq_len, i)
                                 for s, i in zip(finished_globals, ix))

        # End condition is when top-of-beam is EOS and no global score.
        if self.current_state[0] == self._eos:
            self.eos_top = True

    def _successors(self, word_probs):
        beam_scores = word_probs + self.scores.unsqueeze(1)
        # Don't let EOS have children.
        beam_scores[self.current_state == self._eos] = -1e20

        if self.block_ngram_repeat:
            blocked_indices = self._find_ngram_repetitions()
            beam_scores[blocked_indices] = -1e20

        beam_scores = beam_scores.view(-1)
        return beam_scores

    def _find_ngram_repetitions(self):
        n = self.block_ngram_repeat
        le = len(self.next_ys)
        indices = set()
        for j in range(self.width):
            hyp, _ = self.get_hyp(le - 1, j)
            ngrams = set()
            gram = tuple()
            for i in range(le - 1):
                gram = (gram + (hyp[i].item(),))[-n:]
                # Skip the blocking if it is in the exclusion list
                if any(tok in self.exclusion_tokens for tok in gram):
                    continue
                if gram in ngrams:
                    indices.add(j)
                ngrams.add(gram)
        return list(indices)

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=1):
        if len(self.finished) < minimum:
            gs = self.compute_global_score()
            self.finished.extend((gs[i], len(self.next_ys) - 1, i)
                                 for i in range(minimum - len(self.finished)))

        self.finished.sort(reverse=True)
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(timestep - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])

    def update_score(self, attn):
        if "prev_penalty" in self.global_state:
            self.scores.add_(self.global_state["prev_penalty"])
            penalty = self.global_scorer.cov_penalty(
                self, self.global_state["coverage"] + attn,
                self.global_scorer.beta
            )
            self.scores.sub_(penalty)

    def update_global_state(self):
        "Keeps the coverage vector as sum of attentions"
        if len(self.prev_ks) == 1:
            self.global_state["prev_penalty"] = torch.zeros_like(self.scores)
            self.global_state["coverage"] = self.attn[-1]
            self.cov_total = self.attn[-1].sum(1)  # TODO: define elsewhere
        else:
            self.cov_total += torch.min(self.attn[-1],
                                        self.global_state['coverage']).sum(1)
            self.global_state["coverage"] = self.global_state["coverage"] \
                .index_select(0, self.backpointers).add(self.attn[-1])

            prev_penalty = self.global_scorer.cov_penalty(
                self.scores, self.global_state["coverage"]
            )
            self.global_state["prev_penalty"] = prev_penalty

    def compute_global_score(self):
        return self.global_scorer.score(
            self.scores, self.next_ys,
            self.global_state["coverage"],
            self.stepwise_penalty)


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """

    def __init__(self, alpha, beta, cov_penalty, length_penalty):
        self.alpha = alpha
        self.beta = beta
        self._cov_func = penalties.coverage_penalty(cov_penalty)
        self._len_func = penalties.length_penalty(length_penalty)

    def score(self, scores, next_ys, cov, stepwise_penalty=False):
        """
        Rescores a prediction based on penalty functions
        """
        normalized_probs = self.length_penalty(scores, next_ys)
        if not stepwise_penalty:
            penalty = self.cov_penalty(scores, cov)
            normalized_probs -= penalty
        return normalized_probs

    def cov_penalty(self, scores, coverage):
        return self._cov_func(scores, coverage, self.beta)

    def length_penalty(self, scores, next_ys):
        return self._len_func(scores, next_ys, self.alpha)
