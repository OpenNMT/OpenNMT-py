from __future__ import division
import torch


class PenaltyBuilder(object):
    """
    Returns the Length and Coverage Penalty function for Beam Search.

    Args:
        length_pen (str): option name of length pen
        cov_pen (str): option name of cov pen
    """

    def __init__(self, cov_pen, length_pen):
        self.length_pen = length_pen
        self.cov_pen = cov_pen
        self.has_cov_pen = not self._pen_is_none(self.cov_pen)
        self.coverage_penalty = self._coverage_penalty()
        self.has_len_pen = not self._pen_is_none(self.length_pen)
        self.length_penalty = self._length_penalty()

    @staticmethod
    def _pen_is_none(pen):
        return pen == "none" or pen is None

    def _coverage_penalty(self):
        if self.cov_pen == "wu":
            return self.coverage_wu
        elif self.cov_pen == "summary":
            return self.coverage_summary
        elif self._pen_is_none(self.cov_pen):
            return self.coverage_none
        else:
            raise NotImplementedError("No '{:s}' coverage penalty.".format(
                self.cov_pen))

    def _length_penalty(self):
        if self.length_pen == "wu":
            return self.length_wu
        elif self.length_pen == "avg":
            return self.length_average
        elif self._pen_is_none(self.length_pen):
            return self.length_none
        else:
            raise NotImplementedError("No '{:s}' length penalty.".format(
                self.length_pen))

    # Below are all the different penalty terms implemented so far.
    # Subtract coverage penalty from topk log probs.
    # Divide topk log probs by length penalty.

    def coverage_wu(self, cov, beta=0.):
        """GNMT coverage re-ranking score.

        See "Google's Neural Machine Translation System" :cite:`wu2016google`.
        ``cov`` is expected to be sized ``(*, seq_len)``, where ``*`` is
        probably ``batch_size x beam_size`` but could be several
        dimensions like ``batch_size, beam_size``. If ``cov`` is attention,
        then the ``seq_len`` axis probably sums to (almost) 1.
        """

        penalty = -torch.min(cov, cov.clone().fill_(1.0)).log().sum(-1)
        return beta * penalty

    def coverage_summary(self, cov, beta=0.):
        """Our summary penalty."""
        penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(-1)
        penalty -= cov.size(-1)
        return beta * penalty

    def coverage_none(self, cov, beta=0.):
        """Returns zero as penalty"""
        none = torch.zeros((1,), device=cov.device,
                           dtype=torch.float)
        if cov.dim() == 3:
            none = none.unsqueeze(0)
        return none

    def length_wu(self, cur_len, alpha=0.):
        """GNMT length re-ranking score.

        See "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """

        return ((5 + cur_len) / 6.0) ** alpha

    def length_average(self, cur_len, alpha=0.):
        """Returns the current sequence length."""
        return cur_len

    def length_none(self, cur_len, alpha=0.):
        """Returns unmodified scores."""
        return 1.0
