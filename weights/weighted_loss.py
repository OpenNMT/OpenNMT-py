import time
import sys
import math

import torch
import torch.nn as nn
# from torch.nn.modules.loss import NLLLoss, _assert_no_grad
from torch.autograd import Variable

from onmt.Loss import LossComputeBase
import onmt

"""
extend onmt/IO.py to be able to add a property to the targets (the weight), since it's only usefull in training, and it's closely paired with it's value
extend onmt/Loss.py so that i would return a different crit form the actual, crit = nn.NLLLoss(weight, size_average=False)
if there are no datum-weights, it would return crit = nn.NLLLoss(weight, size_average=False, datum_weights=False)

crit returns just a VAR?
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD
"""


class NMTLossComputeDatumWeighted(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, generator, tgt_vocab):
        super(NMTLossComputeDatumWeighted, self).__init__(generator, tgt_vocab)

        self.copy_attn = False
        weight = torch.ones(len(tgt_vocab))
        weight[self.padding_idx] = 0
        self.criterion = DatumWeightedCriterion(weight, size_average=False)

    def compute_loss(self, batch, output, target, datum_weights=None, **kwargs):
        """ See base class for args description. """
        scores = self.generator(self.bottle(output))
        scores_data = scores.data.clone()

        target = target.view(-1)
        target_data = target.data.clone()

        loss = self.criterion(scores, target, datum_weights)
        loss_data = loss.data.clone()

        stats = self.stats(loss_data, scores_data, target_data)

        return loss, stats


class DatumWeightedCriterion(nn.NLLLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, datum_average=False):
        super(DatumWeightedCriterion, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.datum_average = datum_average

    @staticmethod
    def _assert_no_grad(variable):
        assert not variable.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"

    def forward(self, input, target, datum_weights=None):
        self._assert_no_grad(target)
        weights = self._buffers["weight"]

        # for each word (row) in input, i want the n-th value,
        # where n is the value of target[row], meaning the
        # probability of choosing that cat from the model
        conf_logprobs = -torch.squeeze(input.gather(1, target.long().view(-1,1)))

        # for each word (row) in target, i want the value of
        # the weight associated with the cat target[row]
        cat_weights = torch.index_select(weights, 0, target.data.long()).float()

        # Now i produce the weighted prod.
        weighted = conf_logprobs.data * cat_weights

        # do i have datum weights?
        if datum_weights is not None:
            weighted *= datum_weights

        result = weighted.sum()
        if self.size_average:
            result /= cat_weights.sum()

        if self.datum_average:
            result /= datum_weights.sum()

        return torch.autograd.Variable(torch.FloatTensor([result]))



# def DatumWeightedNMTCriterion(vocabSize, opt, pad_id):
#     """
#     Construct a criterion based on the standard NMT Criterion,
#     but which enables to assign a degree of "trustworthiness" to
#     each datum during training
#     """
#     weight = torch.ones(vocabSize)
#     weight[pad_id] = 0
#     # https://github.com/pytorch/pytorch/blob/141f8921ac33270811bc0eb652c1175908045448/torch/nn/modules/loss.py#L64
#     crit = nn.NLLLoss(weight, size_average=False)
#     # crit is a function with returns a Variable "loss", float tensor of size 1 -> torch.autograd.Variable(Tensor(value))
#     # with loss.data being a FloatTensor, which already has the method "clone"
#     # a = torch.FloatTensor([5])
#     # b = torch.autograd.Variable(a)
#     # it's called crit(scores, align, target)
#     # should i rather modify the IO so that i can have the weight on the "targets"?
#
#     if opt.gpuid:
#         crit.cuda()
#     return crit


# class CopyCriterion(object):
#     def __init__(self, vocab_size, force_copy, pad, eps=1e-20):
#         self.force_copy = force_copy
#         self.eps = eps
#         self.offset = vocab_size
#         self.pad = pad
#
#     def __call__(self, scores, align, target):
#         align = align.view(-1)
#
#         # Copy prob.
#         out = scores.gather(1, align.view(-1, 1) + self.offset) \
#                     .view(-1).mul(align.ne(0).float())
#         tmp = scores.gather(1, target.view(-1, 1)).view(-1)
#
#         # Regular prob (no unks and unks that can't be copied)
#         if not self.force_copy:
#             out = out + self.eps + tmp.mul(target.ne(0).float()) + \
#                   tmp.mul(align.eq(0).float()).mul(target.eq(0).float())
#         else:
#             # Forced copy.
#             out = out + self.eps + tmp.mul(align.eq(0).float())
#
#         # Drop padding.
#         loss = -out.log().mul(target.ne(self.pad).float()).sum()
#         return loss
