from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.modules


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
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
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
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

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
    """

    def __init__(self, model,
                 train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 normalization="sents", accum_count=1):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.accum_count = accum_count
        self.padding_idx = self.train_loss.padding_idx
        self.normalization = normalization
        assert(accum_count > 0)
        if accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        idx = 0
        truebatch = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        def gradient_accumulation(truebatch_, total_stats_,
                                  report_stats_, nt_):
            if self.accum_count > 1:
                self.model.zero_grad()

            for batch in truebatch_:
                target_size = batch.tgt.size(0)
                # Truncated BPTT
                if self.trunc_size:
                    trunc_size = self.trunc_size
                else:
                    trunc_size = target_size

                dec_state = None
                src = onmt.io.make_features(batch, 'src', self.data_type)
                if self.data_type == 'text':
                    _, src_lengths = batch.src
                    report_stats.n_src_words += src_lengths.sum()
                else:
                    src_lengths = None

                tgt_outer = onmt.io.make_features(batch, 'tgt')

                for j in range(0, target_size-1, trunc_size):
                    # 1. Create truncated target.
                    tgt = tgt_outer[j: j + trunc_size]

                    # 2. F-prop all but generator.
                    if self.accum_count == 1:
                        self.model.zero_grad()
                    outputs, attns, dec_state = \
                        self.model(src, tgt, src_lengths, dec_state)

                    # 3. Compute loss in shards for memory efficiency.
                    batch_stats = self.train_loss.sharded_compute_loss(
                            batch, outputs, attns, j,
                            trunc_size, self.shard_size, nt_)

                    # 4. Update the parameters and statistics.
                    if self.accum_count == 1:
                        self.optim.step()
                    total_stats_.update(batch_stats)
                    report_stats_.update(batch_stats)

                    # If truncated, don't backprop fully.
                    if dec_state is not None:
                        dec_state.detach()

            if self.accum_count > 1:
                self.optim.step()

        for i, batch_ in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset

            truebatch.append(batch_)
            accum += 1
            if self.normalization is "tokens":
                normalization += batch_.tgt[1:].data.view(-1) \
                                       .ne(self.padding_idx)
            else:
                normalization += batch_.batch_size

            if accum == self.accum_count:
                gradient_accumulation(
                        truebatch, total_stats,
                        report_stats, normalization)

                if report_func is not None:
                    report_stats = report_func(
                            epoch, idx, num_batches,
                            total_stats.start_time, self.optim.lr,
                            report_stats)

                truebatch = []
                accum = 0
                normalization = 0
                idx += 1

        if len(truebatch) > 0:
            gradient_accumulation(
                    truebatch, total_stats,
                    report_stats, normalization)
            truebatch = []

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))
