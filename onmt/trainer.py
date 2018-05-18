"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import onmt.inputters as inputters
import onmt.utils

def build_trainer(opt, model, fields, optim, data_type, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model (after each epoch)
    """
    train_loss = onmt.utils.loss.build_loss_compute(
        model, fields["tgt"].vocab, opt)
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, fields["tgt"].vocab, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count

    report_manager = onmt.utils.build_report_manager(opt)
    trainer = onmt.Trainer(model, train_loss, valid_loss, optim,
                           trunc_size, shard_size, data_type,
                           norm_method, grad_accum_count, report_manager,
                           model_saver=model_saver)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used after each epoch to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1, report_manager=None,
                 model_saver=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.report_manager = report_manager
        self.model_saver = model_saver

        assert grad_accum_count > 0
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter_fct, valid_iter_fct, start_epoch, end_epoch):
        """
        The main training loops.
        It trains from epoch=`start_epoch` to `end_epoch`
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`
        In other words it trains for:
            n_epochs = (end_epoch + 1 - start_epoch, start_epoch)

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            start_epoch(int): epoch number to start from (begining = 0)
            end_epoch(int): last epoch number

        Return:
            None
        """
        print('\nStart training...')
        print(' * number of epochs: %d, starting from Epoch %d' %
              (end_epoch + 1 - start_epoch, start_epoch))
        # print(' * batch size: %d' % batch_size)
        for epoch in range(start_epoch, end_epoch + 1):
            print('')

            # 1. Train for one epoch on the training set.
            train_iter = train_iter_fct()
            train_stats = self.train_epoch(train_iter, epoch)
            self.report_epoch(
                self.optim.learning_rate, epoch, train_stats=train_stats)

            # 2. Validate on the validation set.
            valid_iter = valid_iter_fct()
            valid_stats = self.validate(valid_iter)
            self.report_epoch(
                self.optim.learning_rate, epoch, valid_stats=valid_stats)

            # 3. Update the learning rate
            self.epoch_step(valid_stats.ppl(), epoch)

            # 4. Drop a checkpoint if needed.
            self.maybe_drop_checkpoint(epoch, valid_stats)

    def train_epoch(self, train_iter, epoch):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number

        Returns:
            stats (:obj:`onmt.utils.Statistics`): epoch loss statistics
        """
        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self.start_report_manager(start_time=total_stats.start_time)

        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        for _, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, normalization)

                report_stats = self.report_training(
                    epoch, idx, num_batches,
                    self.optim.learning_rate,
                    report_stats)

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

        if true_batchs:
            self._gradient_accumulation(
                true_batchs, total_stats,
                report_stats, normalization)
            true_batchs = []

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = onmt.utils.Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = inputters.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = inputters.make_features(batch, 'tgt')

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
        """ Epoch step."""
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
            'vocab': inputters.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

    def _gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, normalization):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            src = inputters.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum().item()
            else:
                src_lengths = None

            tgt_outer = inputters.make_features(batch, 'tgt')

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                outputs, attns, dec_state = \
                    self.model(src, tgt, src_lengths, dec_state)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                    batch, outputs, attns, j,
                    trunc_size, self.shard_size, normalization)

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

        if self.grad_accum_count > 1:
            self.optim.step()

    def start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def report_training(self, *args, **kwargs):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(*args, **kwargs)

    def report_epoch(self, *args, **kwargs):
        """
        Simple function to report epoch stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_epoch` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_epoch(*args, **kwargs)

    def maybe_drop_checkpoint(self, *args, **kwargs):
        """
        Drop a checkpoint (i.e. save the model) if a model saver is set
        see `onmt.models.ModelSaverBase.maybe_save` for doc
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(*args, **kwargs)
