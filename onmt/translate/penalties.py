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

    def coverage_penalty(self):
        if self.cov_pen == "wu":
            return self.coverage_wu
        elif self.cov_pen == "summary":
            return self.coverage_summary
        else:
            return self.coverage_none

    def length_penalty(self):
        if self.length_pen == "wu":
            return self.length_wu
        elif self.length_pen == "avg":
            return self.length_average
        else:
            return self.length_none

    # Below are all the different penalty terms implemented so far.
    # Subtract coverage penalty from topk log probs.
    # Divide topk log probs by length penalty.

    def coverage_wu(self, cov, beta=0.):
        """GNMT coverage re-ranking score.

        See "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """

        penalty = -torch.min(cov, cov.clone().fill_(1.0)).log().sum(1)
        return beta * penalty

    def coverage_summary(self, cov, beta=0.):
        """Our summary penalty."""
        penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(1)
        penalty -= cov.size(1)
        return beta * penalty

    def coverage_none(self, cov, beta=0.):
        """Returns zero as penalty"""
        return 0.0

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
