""" Report manager utility """
from __future__ import print_function
import time
from datetime import datetime

import onmt

from onmt.utils.logging import logger


def build_report_manager(opt, gpu_rank):
    if opt.tensorboard and gpu_rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_log_dir = opt.tensorboard_log_dir

        if not opt.train_from:
            tensorboard_log_dir += datetime.now().strftime("/%b-%d_%H-%M-%S")

        writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    else:
        writer = None

    report_mgr = ReportMgr(opt.report_every, start_time=-1,
                           tensorboard_writer=writer)
    return report_mgr


class ReportMgrBase(object):
    """
    Report Manager Base class
    Inherited classes should override:
        * `_report_training`
        * `_report_step`
    """

    def __init__(self, report_every, start_time=-1.):
        """
        Args:
            report_every(int): Report status every this many sentences
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """
        self.report_every = report_every
        self.start_time = start_time

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(self, step, num_steps, learning_rate,
                        report_stats, multigpu=False):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")

        if step % self.report_every == 0:
            if multigpu:
                report_stats = \
                    onmt.utils.Statistics.all_gather_stats(report_stats)
            self._report_training(
                step, num_steps, learning_rate, report_stats)
            if isinstance(report_stats, onmt.utils.BertStatistics):
                return onmt.utils.BertStatistics()
            return onmt.utils.Statistics()
        else:
            return report_stats

    def _report_training(self, *args, **kwargs):
        """ To be overridden """
        raise NotImplementedError()

    def report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        Report stats of a step

        Args:
            train_stats(Statistics): training stats
            valid_stats(Statistics): validation stats
            lr(float): current learning rate
        """
        self._report_step(
            lr, step, train_stats=train_stats, valid_stats=valid_stats)

    def _report_step(self, *args, **kwargs):
        raise NotImplementedError()


class ReportMgr(ReportMgrBase):
    def __init__(self, report_every, start_time=-1., tensorboard_writer=None):
        """
        A report manager that writes statistics on standard output as well as
        (optionally) TensorBoard

        Args:
            report_every(int): Report status every this many sentences
            tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
                The TensorBoard Summary writer to use or None
        """
        super(ReportMgr, self).__init__(report_every, start_time)
        self.tensorboard_writer = tensorboard_writer

    def maybe_log_tensorboard(self, stats, prefix, learning_rate, step):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(
                prefix, self.tensorboard_writer, learning_rate, step)

    def _report_training(self, step, num_steps, learning_rate,
                         report_stats):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output(step, num_steps,
                            learning_rate, self.start_time)

        self.maybe_log_tensorboard(report_stats,
                                   "progress",
                                   learning_rate,
                                   step)
        if isinstance(report_stats, onmt.utils.BertStatistics):
            report_stats = onmt.utils.BertStatistics()
        else:
            report_stats = onmt.utils.Statistics()

        return report_stats

    def _report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None:
            self.log('Train perplexity: %g' % train_stats.ppl())
            if train_stats.accuracy() is None:
                assert isinstance(train_stats, onmt.utils.BertStatistics)
                accuracy = train_stats.sentence_accuracy()
            else:
                accuracy = train_stats.accuracy()
            self.log('Train accuracy: %g' % accuracy)
            if hasattr(train_stats, 'f1') and train_stats.f1 != 0:
                self.log('Train F1: %g' % train_stats.f1)

            self.maybe_log_tensorboard(train_stats,
                                       "train",
                                       lr,
                                       step)

        if valid_stats is not None:
            self.log('Validation perplexity: %g' % valid_stats.ppl())
            if valid_stats.accuracy() is None:
                assert isinstance(valid_stats, onmt.utils.BertStatistics)
                accuracy = valid_stats.sentence_accuracy()
            else:
                accuracy = valid_stats.accuracy()
            self.log('Validation accuracy: %g' % accuracy)
            if hasattr(valid_stats, 'f1') and valid_stats.f1 != 0:
                self.log('Validation F1: %g' % valid_stats.f1)

            self.maybe_log_tensorboard(valid_stats,
                                       "valid",
                                       lr,
                                       step)
