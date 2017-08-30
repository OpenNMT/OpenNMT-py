"""
This file handles the details of the loss function during training.

This includes: loss criterion, training statistics, and memory optimizations.
"""

import onmt
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import sys
import math


def NMTCriterion(vocabSize, opt, pad_id):
    """
    Construct the standard NMT Criterion
    """
    weight = torch.ones(vocabSize)
    weight[pad_id] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpuid:
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
        return 100 * (self.n_correct / float(self.n_words))

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


class Splitter:
    """
    Spliter is a utilty that splits a dictionary of
    data up into shards and waits for them to be backprop'd.
    It blocks until all gradients have been computed and then
    call backward on its inputs.
    """

    def __init__(self, shard_max, eval=False):
        self.shard_max = shard_max
        self.eval = eval

    def splitIter(self, d):
        # If eval mode, don't need to split at all
        if self.eval:
            yield d
            return

        # Split each element and make dummy variable.
        dummies = {}
        shards = []
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, Variable) and v.requires_grad:
                dummies[k] = Variable(v.data, requires_grad=True,
                                      volatile=False)
            else:
                dummies[k] = v
            splits = torch.split(dummies[k], self.shard_max)

            for i, val in enumerate(splits):
                if i >= len(shards):
                    shards.append({})
                shards[i][k] = val

        for i, shard in enumerate(shards):
            yield shard

        # Assumed backprop'd
        inputs = []
        grads = []
        for k, v in dummies.items():
            if isinstance(v, Variable) and (v.grad is not None):
                inputs.append(d[k])
                grads.append(v.grad.data)
        torch.autograd.backward(inputs, grads)
        return


class LossCompute:
    def __init__(self, generator, crit, tgt_vocab, dataset, epoch, opt):
        self.generator = generator
        self.crit = crit
        self.tgt_vocab = tgt_vocab
        self.dataset = dataset
        self.epoch = epoch
        self.opt = opt

    def makeLossBatch(self, outputs, batch, attns, range_):
        """Create all the variables that need to be sharded.
        This needs to match compute loss exactly.
        """
        return {"out": outputs,
                "target": batch.tgt[range_[0] + 1: range_[1]],
                "align": None if not self.opt.copy_attn
                else batch.alignment[range_[0] + 1: range_[1]],
                "coverage": attns.get("coverage"),
                "attn": attns.get("copy")}

    def computeLoss(self, batch, out, target, attn=None,
                    align=None, coverage=None):
        def bottle(v):
            return v.view(-1, v.size(2))

        def unbottle(v):
            return v.view(-1, batch.batch_size, v.size(1))

        pad = self.tgt_vocab.stoi[onmt.IO.PAD_WORD]
        target = target.view(-1)

        if not self.opt.copy_attn:
            # Standard generator.
            scores = self.generator(bottle(out))
            loss = self.crit(scores, target)
            scores_data = scores.data.clone()
            target = target.data.clone()
        else:
            scores = self.generator(bottle(out), bottle(attn), batch.src_map)
            loss = self.crit(scores, align, target)
            scores_data = scores.data.clone()
            scores_data = self.dataset.collapseCopyScores(
                unbottle(scores_data), batch, self.tgt_vocab)
            scores_data = bottle(scores_data)

            # Correct target is copy when only option.
            target = target.data.clone()
            for i in range(target.size(0)):
                if target[i] == 0 and align.data[i] != 0:
                    target[i] = align.data[i] + len(self.tgt_vocab)

        # Coverage loss term.
        ppl = loss.data.clone()

        stats = Statistics.score(ppl, scores_data, target, pad)
        return loss, stats
