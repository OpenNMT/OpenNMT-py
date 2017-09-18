import time
import sys
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt


def DatumWeightedNMTCriterion(vocabSize, opt, pad_id):
    """
    Construct a criterion based on the standard NMT Criterion,
    but which enables to assign a degree of "trustworthiness" to
    each datum during training
    """
    weight = torch.ones(vocabSize)
    weight[pad_id] = 0
    # https://github.com/pytorch/pytorch/blob/141f8921ac33270811bc0eb652c1175908045448/torch/nn/modules/loss.py#L64
    crit = nn.NLLLoss(weight, size_average=False)
    # crit is a function with returns a Variable "loss", float tensor of size 1 -> torch.autograd.Variable(Tensor(value))
    # with loss.data being a FloatTensor, which already has the method "clone"
    # a = torch.FloatTensor([5])
    # b = torch.autograd.Variable(a)
    # it's called crit(scores, align, target)
    # should i rather modify the IO so that i can have the weight on the "targets"?

    if opt.gpuid:
        crit.cuda()
    return crit


class CopyCriterion(object):
    def __init__(self, vocab_size, force_copy, pad, eps=1e-20):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = vocab_size
        self.pad = pad

    def __call__(self, scores, align, target):
        align = align.view(-1)

        # Copy prob.
        out = scores.gather(1, align.view(-1, 1) + self.offset) \
                    .view(-1).mul(align.ne(0).float())
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)

        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            out = out + self.eps + tmp.mul(target.ne(0).float()) + \
                  tmp.mul(align.eq(0).float()).mul(target.eq(0).float())
        else:
            # Forced copy.
            out = out + self.eps + tmp.mul(align.eq(0).float())

        # Drop padding.
        loss = -out.log().mul(target.ne(self.pad).float()).sum()
        return loss


# Standard calculation of loss
# from torch/nn/_functions/thnn/auto_double_backwards.py


def nllloss_double_backwards(ctx, ggI):
    t = ctx.saved_variables
    target = t[1]
    weights = Variable(ctx.additional_args[1])
    size_average = ctx.additional_args[0]
    ignore_index = ctx.additional_args[3]

    gI = None

    # can't scatter/gather on indices outside of range, let's just put them in range
    # and 0 out the weights later (so it doesn't matter where in range we put them)
    target_mask = target == ignore_index
    safe_target = target.clone()
    safe_target.masked_fill_(target_mask, 0)

    if weights.dim() == 0:
        weights_to_scatter = Variable(ggI.data.new(safe_target.size()).fill_(1))
    else:
        weights_maybe_resized = weights
        while weights_maybe_resized.dim() < target.dim():
            weights_maybe_resized = weights_maybe_resized.unsqueeze(1)

        weights_maybe_resized = weights_maybe_resized.expand(weights.size()[0:1] + target.size()[1:])
        weights_to_scatter = weights_maybe_resized.gather(0, safe_target)

    weights_to_scatter.masked_fill_(target_mask, 0)
    divisor = weights_to_scatter.sum() if size_average else 1
    weights_to_scatter = -1 * weights_to_scatter / divisor
    zeros = Variable(ggI.data.new(ggI.size()).zero_())
    mask = zeros.scatter_(1, safe_target.unsqueeze(1), weights_to_scatter.unsqueeze(1))

    ggO = (ggI * mask).sum()

    return gI, None, ggO, None, None, None