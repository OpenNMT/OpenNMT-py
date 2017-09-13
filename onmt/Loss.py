"""
This file handles the details of the loss function during training.

This includes: loss criterion, training statistics, and memory optimizations.
"""
from __future__ import division
import time
import sys
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt


def nmt_criterion(vocab_size, gpuid, pad_id):
    """
    Construct the standard NMT Criterion
    """
    weight = torch.ones(vocab_size)
    weight[pad_id] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if gpuid:
        crit.cuda()
    return crit


class Statistics:
    """
    Training loss function statistics.
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, optim):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", optim.lr)

    @staticmethod
    def score(loss, scores, targ, pad):
        pred = scores.max(1)[1]
        non_padding = targ.ne(pad)
        num_correct = pred.eq(targ) \
                          .masked_select(non_padding) \
                          .sum()
        return Statistics(loss[0], non_padding.sum(), num_correct)


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


class LossCompute:
    def __init__(self, generator, crit, tgt_vocab, dataset, epoch, copy_attn):
        self.generator = generator
        self.crit = crit
        self.tgt_vocab = tgt_vocab
        self.dataset = dataset
        self.epoch = epoch
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

        stats = Statistics.score(ppl, scores_data, target, pad)
        return loss, stats
