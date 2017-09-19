"""
This file handles the details of the loss function during training.

This includes: loss computation and memory optimizations(shard).
"""
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

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


def nmt_criterion(vocab_size, gpuid, pad_id):
    """
    Standard NMT Loss Criterion.
    """
    weight = torch.ones(vocab_size)
    weight[pad_id] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if gpuid:
        crit.cuda()
    return crit


def make_gen_state(output, batch, attns, range_, copy_attn=None):
    """
    Create generator state for use in sharded loss computation.
    This needs to match compute_loss exactly.
    """
    return {"output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "align": None if not copy_attn
            else batch.alignment[range_[0] + 1: range_[1]],
            "coverage": attns.get("coverage"),
            "copy_attn": attns.get("copy")}


def filter_gen_state(state):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=True, volatile=False)
            yield k, v


def shards(state, shard_size, eval=False):
    """
    state:
        A dictionary which corresponds to the output of
        LossCompute.make_loss_batch(). In other words, its keys are
        {'out', 'target', 'align', 'coverage', 'attn'}. The values
        for those keys are Tensor-like or None.
    shard_size:
        The maximum size of the shards yielded by the model
    eval:
        If True, only yield the state, nothing else. Otherwise, yield shards.
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


class LossCompute(object):
    """
    This is the loss criterion object, wrapping around the normal pytorch
    *Loss criterion or user-defined criterion, with some project specified
    attributes like copy attention, etc.
    """
    def __init__(self, generator, crit, tgt_vocab, dataset, copy_attn):
        self.generator = generator
        self.crit = crit
        self.tgt_vocab = tgt_vocab
        self.dataset = dataset
        self.copy_attn = copy_attn

    def make_loss_batch(self, outputs, batch, attns, range_):
        """
        Create all the variables that need to be sharded.
        This needs to match compute loss exactly.
        """
        return {"out": outputs,
                "target": batch.tgt[range_[0] + 1: range_[1]],
                "align": None if not self.copy_attn
                else batch.alignment[range_[0] + 1: range_[1]],
                "coverage": attns.get("coverage"),
                "attn": attns.get("copy")}

    def compute_loss(self, batch, out, target, attn=None,
                     align=None, coverage=None):
        def bottle(v):
            return v.view(-1, v.size(2))

        def unbottle(v):
            return v.view(-1, batch.batch_size, v.size(1))

        pad = self.tgt_vocab.stoi[onmt.IO.PAD_WORD]
        target = target.view(-1)

        if not self.copy_attn:
            # Standard generator.
            scores = self.generator(bottle(out))
            loss = self.crit(scores, target)
            scores_data = scores.data.clone()
            target = target.data.clone()
        else:
            align = align.view(-1)
            scores = self.generator(bottle(out), bottle(attn), batch.src_map)
            loss = self.crit(scores, align, target)
            scores_data = scores.data.clone()
            scores_data = self.dataset.collapse_copy_scores(
                unbottle(scores_data), batch, self.tgt_vocab)
            scores_data = bottle(scores_data)

            # Correct target is copy when only option.
            # TODO: replace for loop with masking or boolean indexing
            target = target.data.clone()
            for i in range(target.size(0)):
                if target[i] == 0 and align.data[i] != 0:
                    target[i] = align.data[i] + len(self.tgt_vocab)

        # Coverage loss term.
        ppl = loss.data.clone()

        stats = self.score(ppl, scores_data, target, pad)

        return loss, stats

    def score(self, loss, scores, targ, pad):
        pred = scores.max(1)[1]
        non_padding = targ.ne(pad)
        num_correct = pred.eq(targ) \
                          .masked_select(non_padding) \
                          .sum()
        return onmt.Statistics(loss[0], non_padding.sum(), num_correct)
