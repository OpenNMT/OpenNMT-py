from __future__ import division
import torch


def coverage_penalty(name):
    coverages = {"wu": coverage_wu, "summary": coverage_summary}
    return coverages.get(name, coverage_none)


def length_penalty(name):
    lengths = {"wu": length_wu, "average": length_average}
    return lengths.get(name, length_none)


def coverage_wu(scores, cov, beta=0.):
    """
    NMT coverage re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """
    return 0. if beta == 0. else beta * -cov.clamp(max=1.0).log().sum(1)


def coverage_summary(scores, cov, beta=0.):
    """
    Our summary penalty.
    """
    return 0. if beta == 0. else beta * \
        (-cov.clamp(min=1.0).sum(1) - cov.size(1))


def coverage_none(scores, cov, beta=0.):
    """
    returns zero as penalty
    """
    return 0.0


def length_wu(scores, next_ys, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """
    modifier = ((5 + len(next_ys)) / 6) ** alpha
    return scores / modifier


def length_average(scores, next_ys, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return scores / len(next_ys)


def length_none(scores, next_ys, alpha=0.):
    """
    Returns unmodified scores.
    """
    return scores
