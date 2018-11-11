"""
An implementation of sparsemax (Martins & Astudillo, 2016). See
https://arxiv.org/pdf/1602.02068 for detailed description.

By Ben Peters and Vlad Niculae
"""

import torch
from torch.autograd import Function
import torch.nn as nn


def _make_ix_like(X, dim=0):
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _threshold_and_support(X, dim=0):
    """
    Sparsemax building block: compute the threshold
    Parameters:
        X: any dimension
        dim: dimension along which to apply the sparsemax
    Returns:
        the threshold value
    """
    X_srt, _ = torch.sort(X, descending=True, dim=dim)
    X_cumsum = X_srt.cumsum(dim) - 1
    rhos = _make_ix_like(X, dim)
    support = rhos * X_srt > X_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = X_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(X.dtype)
    return tau, support_size


class SparsemaxFunction(Function):

    @staticmethod
    def forward(ctx, X, dim=0):
        """
        sparsemax: normalizing sparse transform (a la softmax)
        Parameters:
            X (Tensor): any shape
            dim: dimension along which to apply sparsemax
        Returns:
            Y (Tensor): same shape as X
        """
        ctx.dim = dim
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as for softmax
        tau, support_size = _threshold_and_support(X, dim=dim)
        Y = torch.clamp(X - tau, min=0)
        ctx.save_for_backward(support_size, Y)
        return Y

    @staticmethod
    def backward(ctx, dY):
        support_size, Y = ctx.saved_tensors
        dim = ctx.dim
        dX = dY.clone()
        dX[Y == 0] = 0

        v_hat = dX.sum(dim=dim) / support_size.to(Y.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        dX = torch.where(Y != 0, dX - v_hat, dX)
        return dX, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


class LogSparsemax(nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(LogSparsemax, self).__init__()

    def forward(self, input):
        return torch.log(sparsemax(input, self.dim))
