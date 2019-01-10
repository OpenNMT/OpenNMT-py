""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys

from torch.distributed import get_rank
from onmt.utils.distributed import all_gather_list
from onmt.utils.logging import logger
from collections import Counter


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * bleu
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0,
                 precision_matches=Counter(), precision_totals=Counter(),
                 prediction_lengths=0, reference_lengths=0,
                 ngram_weights=(0.25, 0.25, 0.25, 0.25),
                 exclude_indices=None):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self._ngram_weights = ngram_weights
        self._exclude_indices = exclude_indices or set()
        self.precision_matches = precision_matches
        self.precision_totals = precision_totals
        self.prediction_lengths = prediction_lengths
        self.reference_lengths = reference_lengths
        self.start_time = time.time()

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.precision_matches = {key:
                                  self.precision_matches.get(key, 0)
                                  + stat.precision_matches.get(key, 0)
                                  for key in
                                  set(self.precision_matches) |
                                  set(stat.precision_matches)}

        self.precision_totals = {key:
                                 self.precision_totals.get(key, 0)
                                 + stat.precision_totals.get(key, 0)
                                 for key in
                                 set(self.precision_totals) |
                                 set(stat.precision_totals)}
        self.prediction_lengths += stat.prediction_lengths
        self.reference_lengths += stat.reference_lengths
        if update_n_src_words:
            self.n_src_words += stat.n_src_words

    def _get_brevity_penalty(self):
        if self.prediction_lengths > self.reference_lengths:
            return 1.0
        if self.reference_lengths == 0 or self.prediction_lengths == 0:
            return 0.0
        return math.exp(1.0 - self.reference_lengths /
                        self.prediction_lengths)

    def bleu(self):
        brevity_penalty = self._get_brevity_penalty()
        ngram_scores = (weight *
                        (math.log(self.precision_matches[n] + 1e-13) -
                            math.log(self.precision_totals[n] + 1e-13))
                        for n, weight in
                        enumerate(self._ngram_weights, start=1))
        bleu = brevity_penalty * math.exp(sum(ngram_scores))
        return bleu

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        logger.info(
            ("Step %2d/%5d; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
             "lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
            % (step, num_steps,
               self.accuracy(),
               self.ppl(),
               self.xent(),
               learning_rate,
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
