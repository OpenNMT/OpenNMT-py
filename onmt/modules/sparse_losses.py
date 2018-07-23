import torch
import torch.nn as nn
from torch.autograd import Function
from onmt.modules.sparse_activations import threshold_and_support


class SparsemaxLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target):
        """
        input (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        z_k = input.gather(1, target.unsqueeze(1)).squeeze()
        tau_z, support_size = threshold_and_support(input)
        support = input > tau_z
        x = torch.where(
            support, input**2 - tau_z**2, torch.tensor(0.0)
        ).sum(dim=1)
        ctx.save_for_backward(input, target, tau_z)
        return x / 2 - z_k + 0.5

    @staticmethod
    def backward(ctx, grad_output):
        input, target, tau_z = ctx.saved_tensors
        sparsemax_out = torch.clamp(input - tau_z, min=0)
        delta = torch.zeros_like(sparsemax_out)
        delta.scatter_(1, target.unsqueeze(1), 1)
        return sparsemax_out - delta, None


sparsemax_loss = SparsemaxLossFunction.apply


class SparsemaxLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=-100,
                 reduce=True, size_average=True):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.size_average = size_average
        super(SparsemaxLoss, self).__init__()

    def forward(self, input, target):
        loss = sparsemax_loss(input, target)
        if self.ignore_index >= 0:
            ignored_positions = target == self.ignore_index
            size = float((target.size(0) - ignored_positions.sum()).item())
            loss.masked_fill_(ignored_positions, 0.0)
        else:
            size = float(target.size(0))
        if self.reduce:
            loss = loss.sum()
            if self.size_average:
                loss = loss / size
        return loss
