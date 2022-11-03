
"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

import torch
import traceback
import onmt.utils
from onmt.utils.loss import LossCompute
from onmt.utils.logging import logger
from onmt.translate.utils import ScoringPreparator
from onmt.scorers import get_scorers_cls, build_scorers


def build_trainer(opt, device_id, model, vocabs, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    train_loss = LossCompute.from_opts(opt, model, vocabs['tgt'])
    valid_loss = LossCompute.from_opts(opt, model, vocabs['tgt'], train=False)

    scoring_preparator = ScoringPreparator(vocabs, opt)
    scorers_cls = get_scorers_cls(opt.train_metrics)
    train_scorers = build_scorers(opt, scorers_cls)
    scorers_cls = get_scorers_cls(opt.valid_metrics)
    valid_scorers = build_scorers(opt, scorers_cls)

    trunc_size = opt.truncated_decoder  # Badly named...
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    attention_dropout = opt.attention_dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = -1
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    report_manager = onmt.utils.build_report_manager(opt, gpu_rank)
    trainer = onmt.Trainer(model,
                           train_loss, valid_loss,
                           scoring_preparator, train_scorers, valid_scorers,
                           optim, trunc_size,
                           accum_count, accum_steps,
                           n_gpu, gpu_rank, gpu_verbose_level,
                           opt.train_eval_steps, report_manager,
                           with_align=True if opt.lambda_align > 0 else False,
                           model_saver=model_saver if gpu_rank <= 0 else None,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper,
                           dropout=dropout,
                           attention_dropout=attention_dropout,
                           dropout_steps=dropout_steps)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.
    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            data_type(string): type of the source input: [text]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss,
                 scoring_preparator, train_scorers, valid_scorers,
                 optim,
                 trunc_size=0,
                 accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1, gpu_verbose_level=0,
                 train_eval_steps=200,
                 report_manager=None, with_align=False, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], attention_dropout=[0.1],
                 dropout_steps=[0]):
        # Basic attributes.

        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss

        self.scoring_preparator = scoring_preparator
        self.train_scorers = train_scorers
        self.valid_scorers = valid_scorers
        self.optim = optim
        self.trunc_size = trunc_size
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.train_eval_steps = train_eval_steps
        self.with_align = with_align
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.dropout_steps = dropout_steps

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0

        # Set model in training mode.
        self.model.train()

    def training_eval_handler(self, scorer, batch, mode="train"):
        """Trigger metrics calculations"""
        preds, texts_ref = self.scoring_preparator.translate(
            model=self.model,
            batch=batch,
            gpu_rank=self.gpu_rank,
            step=self.optim.training_step,
            mode=mode)
        return scorer.compute_score(preds, texts_ref)

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i],
                                          self.attention_dropout[i])
                logger.info("Updated dropout/attn dropout to %f %f at step %d"
                            % (self.dropout[i],
                               self.attention_dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            if len(batches) == self.accum_count:
                yield batches
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
        if batches:
            yield batches

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1) / (step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.
        Returns:
            The gathered statistics.
        """
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        for i, batches in enumerate(
                self._accum_batches(train_iter)):
            step = self.optim.training_step
            # UPDATE DROPOUT
            self._maybe_update_dropout(step)

            self._gradient_accumulation(
                batches, total_stats,
                report_stats)

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(
                step, train_steps,
                self.optim.learning_rate(),
                report_stats)

            if (valid_iter is not None and step % valid_steps == 0 and
                    self.gpu_rank == 0):

                valid_stats = self.validate(
                    valid_iter, moving_average=self.moving_average)
                self._report_step(self.optim.learning_rate(),
                                  step, train_stats=total_stats,
                                  valid_stats=valid_stats)

                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        logger.info("earlystopper has_stopped!")
                        break

            if (self.model_saver is not None
                    and (save_checkpoint_steps != 0
                         and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats

    def validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" \
                    else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src = batch['src']
                src_len = batch['srclen']
                tgt = batch['tgt']

                with torch.cuda.amp.autocast(enabled=self.optim.amp):
                    # F-prop through the model.
                    model_out, attns = valid_model(src, tgt, src_len,
                                                   with_align=self.with_align)

                    # Compute loss.
                    _, batch_stats = self.valid_loss(batch, model_out, attns)

                    stats.update(batch_stats)

            # Compute validation metrics (at batch.dataset level)
            computed_metrics = {}
            for i, metric in enumerate(self.valid_scorers):
                logger.info("UPDATING VALIDATION {}".format(metric))
                self.valid_scorers[
                    metric]["value"] = self.training_eval_handler(
                        scorer=self.valid_scorers[metric]["scorer"],
                        batch=batch,
                        mode="valid")
                computed_metrics[
                    metric] = self.valid_scorers[metric]["value"]
                logger.info(
                    "validation {}: {}".format(
                        metric, self.valid_scorers[metric]["value"])
                        )
                # Compute stats
                batch_stats = onmt.utils.Statistics(
                    batch_stats.loss,
                    batch_stats.n_words,
                    batch_stats.n_correct,
                    computed_metrics)

                # Update statistics.
                stats.update(batch_stats)

        if moving_average:
            for param_data, param in zip(model_params_data,
                                         self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

        return stats

    def _gradient_accumulation(self, true_batches, total_stats,
                               report_stats):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):
            target_size = batch['tgt'].size(1)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            src = batch['src']
            src_len = batch['srclen']
            if src_len is not None:
                report_stats.n_src_words += src_len.sum().item()
                total_stats.n_src_words += src_len.sum().item()

            tgt_outer = batch['tgt']

            bptt = False
            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.

                tgt = tgt_outer[:, j: j + trunc_size, :]

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad()

                try:
                    with torch.cuda.amp.autocast(enabled=self.optim.amp):
                        model_out, attns = self.model(
                            src, tgt, src_len, bptt=bptt,
                            with_align=self.with_align)
                        bptt = True

                        # 3. Compute loss.
                        loss, batch_stats = self.train_loss(
                            batch,
                            model_out,
                            attns,
                            trunc_start=j,
                            trunc_size=trunc_size)

                    step = self.optim.training_step
                    if (
                            step % self.train_eval_steps == 0 and
                            self.n_gpu > 0
                    ):
                        # Compute and save stats
                        computed_metrics = {}
                        for i, metric in enumerate(self.train_scorers):
                            logger.info("UPDATING TRAINING {}".format(metric))
                            self.train_scorers[
                                metric]["value"] = self.training_eval_handler(
                                scorer=self.train_scorers[
                                    metric]["scorer"],
                                batch=batch,
                                mode="train")
                            logger.info(
                                "training {}: {}".format(
                                    metric, self.train_scorers[
                                        metric]["value"]))
                            computed_metrics[
                                metric] = self.train_scorers[metric]["value"]
                        batch_stats.computed_metrics = computed_metrics

                    if loss is not None:
                        # in theory we should divide by accum_count and bptt
                        # to rescale for each sub batch
                        self.optim.backward(loss)

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                except Exception as exc:
                    trace_content = traceback.format_exc()
                    if "CUDA out of memory" in trace_content:
                        logger.info("Step %d, cuda OOM - batch removed",
                                    self.optim.training_step)
                    else:
                        traceback.print_exc()
                        raise exc

                # 4. Update the parameters and statistics.
                if self.accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [p.grad.data for p in self.model.parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(self.n_gpu))
                    self.optim.step()

                # If truncated, don't backprop fully.
                if self.model.decoder.state != {}:
                    self.model.decoder.detach_state()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(self.n_gpu))
            self.optim.step()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step,
                num_steps,
                learning_rate,
                None if self.earlystopper is None
                else self.earlystopper.current_tolerance,
                report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate,
                None if self.earlystopper is None
                else self.earlystopper.current_tolerance,
                step, train_stats=train_stats,
                valid_stats=valid_stats)
