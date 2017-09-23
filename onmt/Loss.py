"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import NLLLoss

import onmt


class LossComputeBase(nn.Module):
    """
    This is the loss criterion base class. Users can implement their own
    loss computation strategy by making subclass of this one.
    Users need to implement the compute_loss() method.
    We inherits from nn.Module to leverage the cuda behavior.
    """
    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[onmt.IO.PAD_WORD]

    def forward(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define the compute_loss().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs: additional info for computing loss.
        """
        # Need to simplify this interface.
        return self.compute_loss(batch, output, target, **kwargs)

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size):
        """
        Compute the loss in shards for efficiency.
        """
        batch_stats = onmt.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        gen_state = make_gen_state(output, batch, attns, range_,
                                   self.copy_attn)

        for shard in shards(gen_state, shard_size):
            loss, stats = self.compute_loss(batch, **shard)
            loss.div(batch.batch_size).backward()
            batch_stats.update(stats)

        return batch_stats

    def stats(self, loss, scores, target):
        """
        Compute and return a Statistics object.

        Args:
            loss(Tensor): the loss computed by the loss criterion.
            scores(Tensor): a sequence of predict output with scores.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum()
        return onmt.Statistics(loss[0], non_padding.sum(), num_correct)

    def bottle(self, v):
        return v.view(-1, v.size(2))

    def unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, generator, tgt_vocab):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)

        self.copy_attn = False
        weight = torch.ones(len(tgt_vocab))
        weight[self.padding_idx] = 0
        self.criterion = nn.NLLLoss(weight, size_average=False)

    def compute_loss(self, batch, output, target, **kwargs):
        """ See base class for args description. """
        scores = self.generator(self.bottle(output))
        scores_data = scores.data.clone()

        target = target.view(-1)
        target_data = target.data.clone()

        loss = self.criterion(scores, target)
        loss_data = loss.data.clone()

        stats = self.stats(loss_data, scores_data, target_data)

        return loss, stats


class DatumWeightedNMTLossCompute(LossComputeBase):
    """
    An extension of the Standard NMT Loss Computation that
    includes the possibility to use independent weights for
    every datum
    """
    def __init__(self, generator, tgt_vocab):
        super(DatumWeightedNMTLossCompute, self).__init__(generator, tgt_vocab)
        self.copy_attn = False
        weight = torch.ones(len(tgt_vocab))
        weight[self.padding_idx] = 0
        self.criterion = DatumWeightedNLLCriterion(weight, size_average=False)

    def compute_loss(self, batch, output, target, dw):
        """ See base class for args description. """
        scores = self.generator(self.bottle(output))
        scores_data = scores.data.clone()

        # since i have only one value per target, i'll need to cope with target
        if len(target.size()) > 1:
            tgt_dims = target.size()[1]
            dw_for_view = torch.stack([dw for _ in range(tgt_dims)],
                                      1)

        target = target.view(-1)
        target_data = target.data.clone()

        dw_for_view = dw_for_view.view(-1)
        # dw_for_view_data = dw_for_view.data.clone()

        loss = self.criterion(scores, target, datum_weights=dw_for_view)
        loss_data = loss.data.clone()

        stats = self.stats(loss_data, scores_data, target_data)

        return loss, stats


class DatumWeightedNLLCriterion(NLLLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100,
                 datum_average=False):
        super(NLLLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.datum_average = datum_average

    def _assert_no_grad(self, variable):
        assert not variable.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - " \
            "please mark these variables as volatile or not requiring" \
            "gradients"

    def forward(self, input, target, datum_weights=None):
        self._assert_no_grad(target)
        weights = self._buffers["weight"]

        # for each word (row) in input, i want the n-th value,
        # where n is the value of target[row], meaning the
        # probability of choosing that cat from the model
        conf_logprobs = -torch.squeeze(input.gather(1, target.view(-1, 1)))

        # for each word (row) in target, i want the value of
        # the weight associated with the cat target[row]
        cat_weights = torch.autograd.Variable(
            torch.index_select(weights, 0, target.data))

        # Now i produce the weighted prod.
        weighted = conf_logprobs * cat_weights

        # do i have datum weights?
        if datum_weights is not None:
            weighted = weighted * datum_weights

        result = weighted.sum()
        divisors = torch.autograd.Variable(torch.FloatTensor([1.0]))
        if self.size_average:
            divisors = divisors * cat_weights.sum()

        if self.datum_average:
            divisors = divisors * datum_weights.sum()

        # divisors = torch.autograd.Variable(torch.FloatTensor([divisors]))
        result = result / divisors
        return result


def make_gen_state(output, batch, attns, range_, copy_attn=None):
    """
    Create generator state for use in sharded loss computation.
    This needs to match compute_loss exactly.
    """
    if copy_attn and getattr(batch, 'alignment', None) is None:
        raise AssertionError("using -copy_attn you need to pass in "
                             "-dynamic_dict during preprocess stage.")

    use_dw = hasattr(batch, "dw")
    return {"output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "copy_attn": attns.get("copy"),
            "align": None if not copy_attn
            else batch.alignment[range_[0] + 1: range_[1]],
            "coverage": attns.get("coverage"),
            "dw": None if not use_dw
            else batch.dw[range_[0] + 1: range_[1]]}


def filter_gen_state(state):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=True, volatile=False)
            yield k, v


def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               make_gen_state(). The values for those keys are
               Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    yields:
        Each yielded shard is a dict.
    side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield state
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_gen_state(state))

        # Now, the iteration:
        # split_state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, torch.split(v, shard_size))
                             for k, v in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
