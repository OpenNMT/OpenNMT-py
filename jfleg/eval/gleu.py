#!/usr/bin/env python
"""
(Note: This script computes sentence-level GLEU score.)

This script calculates the GLEU score of a sentence, as described in
our ACL 2015 paper, Ground Truth for Grammatical Error Correction Metrics
by Courtney Napoles, Keisuke Sakaguchi, Matt Post, and Joel Tetreault.

For instructions on how to get the GLEU score, call "compute_gleu -h"

Updated 2 May 2016: This is an updated version of GLEU that has been
modified to handle multiple references more fairly.

This script was adapted from bleu.py by Adam Lopez.
<https://github.com/alopez/en600.468/blob/master/reranker/>
"""
__author__ = 'Courtney Napoles'
__email__ = 'napoles@cs.jhu.edu'
__date__ = '2016-11-04'

import math
import numpy as np
import scipy.stats
import sys
import random
import argparse
from collections import Counter


class GLEU:

    def __init__(self, n=4):
        self.order = n

    def load_hypothesis_sentence(self, hypothesis):
        """load ngrams for a single sentence"""
        self.hlen = len(hypothesis)
        self.this_h_ngrams = [self.get_ngram_counts(hypothesis, n)
                              for n in range(1, self.order + 1)]

    def load_sources(self, spath):
        """load n-grams for all source sentences"""
        self.all_s_ngrams = [[self.get_ngram_counts(line.split(), n)
                              for n in range(1, self.order + 1)]
                             for line in open(spath)]

    def load_references(self, rpaths):
        """load n-grams for all references"""
        self.refs = [[] for i in range(len(self.all_s_ngrams))]
        self.rlens = [[] for i in range(len(self.all_s_ngrams))]
        self.num_refs = len(rpaths)
        for rpath in rpaths:
            for i, line in enumerate(open(rpath)):
                self.refs[i].append(line.split())
                self.rlens[i].append(len(line.split()))

        # count number of references each n-gram appear sin
        self.all_rngrams_freq = [Counter() for i in range(self.order)]

        self.all_r_ngrams = []
        for refset in self.refs:
            all_ngrams = []
            self.all_r_ngrams.append(all_ngrams)

            for n in range(1, self.order + 1):
                ngrams = self.get_ngram_counts(refset[0], n)
                all_ngrams.append(ngrams)

                for k in ngrams.keys():
                    self.all_rngrams_freq[n - 1][k] += 1

                for ref in refset[1:]:
                    new_ngrams = self.get_ngram_counts(ref, n)
                    for nn in new_ngrams.elements():
                        if new_ngrams[nn] > ngrams.get(nn, 0):
                            ngrams[nn] = new_ngrams[nn]

    def get_ngram_counts(self, sentence, n):
        """get ngrams of order n for a tokenized sentence"""
        return Counter([tuple(sentence[i:i + n])
                        for i in range(len(sentence) + 1 - n)])

    def get_ngram_diff(self, a, b):
        """returns ngrams in a but not in b"""
        diff = Counter(a)
        for k in (set(a) & set(b)):
            del diff[k]
        return diff

    def normalization(self, ngram, n):
        """get normalized n-gram count"""
        return 1.0 * self.all_rngrams_freq[n - 1][ngram] / len(self.rlens[0])

    def gleu_stats(self, i, r_ind=None):
        """
        Collect BLEU-relevant statistics for a single hypothesis/reference pair.
        Return value is a generator yielding:
        (c, r, numerator1, denominator1, ... numerator4, denominator4)
        Summing the columns across calls to this function on an entire corpus
        will produce a vector of statistics that can be used to compute GLEU
        """
        hlen = self.hlen
        rlen = self.rlens[i][r_ind]

        yield hlen
        yield rlen

        for n in range(1, self.order + 1):
            h_ngrams = self.this_h_ngrams[n - 1]
            s_ngrams = self.all_s_ngrams[i][n - 1]
            r_ngrams = self.get_ngram_counts(self.refs[i][r_ind], n)

            s_ngram_diff = self.get_ngram_diff(s_ngrams, r_ngrams)

            yield max(
                [sum((h_ngrams & r_ngrams).values()) - sum((h_ngrams & s_ngram_diff).values()), 0])

            yield max([hlen + 1 - n, 0])

    def gleu(self, stats, smooth=False):
        """Compute GLEU from collected statistics obtained by call(s) to gleu_stats"""
        # smooth 0 counts for sentence-level scores
        if smooth:
            stats = [s if s != 0 else 1 for s in stats]
        if len(list(filter(lambda x: x == 0, stats))) > 0:
            return 0
        (c, r) = stats[:2]
        log_gleu_prec = sum([math.log(float(x) / y)
                             for x, y in zip(stats[2::2], stats[3::2])]) / 4
        return math.exp(min([0, 1 - float(r) / c]) + log_gleu_prec)

    def get_gleu_stats(self, scores):
        """calculate mean and confidence interval from all GLEU iterations"""
        mean = np.mean(scores)
        std = np.std(scores)
        ci = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
        return ['%f' % mean,
                '%f' % std,
                '(%.3f,%.3f)' % (ci[0], ci[1])]

    def run_iterations(self, num_iterations=500, n=4, source='source.text',
                       hypothesis='answer.txt',
                       debug=False, per_sent=True):
        """run specified number of iterations of GLEU, choosing a reference
        for each sentence at random"""

        instream = sys.stdin if hypothesis == '-' else open(hypothesis)
        hyp = [line.split() for line in instream]

        # first generate a random list of indices, using a different seed
        # for each iteration
        indices = []
        for j in range(num_iterations):
            random.seed(j * 101)
            indices.append([random.randint(0, self.num_refs - 1)
                            for i in range(len(hyp))])

        if debug:
            print('===== Sentence-level scores =====')
            print('SID Mean Stdev 95%CI GLEU')

        iter_stats = [[0 for i in range(2 * n + 2)] for j in range(num_iterations)]

        for i, h in enumerate(hyp):

            self.load_hypothesis_sentence(h)
            # we are going to store the score of this sentence for each ref
            # so we don't have to recalculate them 500 times

            stats_by_ref = [None for r in range(self.num_refs)]

            for j in range(num_iterations):
                ref = indices[j][i]
                this_stats = stats_by_ref[ref]

                if this_stats is None:
                    this_stats = [s for s in self.gleu_stats(i, r_ind=ref)]
                    stats_by_ref[ref] = this_stats

                iter_stats[j] = [sum(scores) for scores in zip(iter_stats[j], this_stats)]

            if debug or per_sent:
                # sentence-level GLEU is the mean GLEU of the hypothesis
                # compared to each reference
                for r in range(self.num_refs):
                    if stats_by_ref[r] is None:
                        stats_by_ref[r] = [s for s in self.gleu_stats(i, r_ind=r)]
                if debug:
                    print(i, ' '.join(
                        self.get_gleu_stats([self.gleu(stats, smooth=True)
                                             for stats in stats_by_ref])))
                yield self.get_gleu_stats([self.gleu(stats, smooth=True)
                                           for stats in stats_by_ref])
        if not per_sent:
            yield self.get_gleu_stats([self.gleu(stats) for stats in iter_stats])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', '-r', nargs='*', required=True, help='ref file(s)')
    parser.add_argument('--src', '-s', required=True, help='src file')
    parser.add_argument('--hyp', nargs='*', required=True, help='hyp file(s)')
    parser.add_argument('-n', default=4, type=int, help='n-gram order')
    parser.add_argument('--iter', default=500, help='number of GLEU iterations')
    parser.add_argument('--sent', default=False, action='store_true', help='sentence level scores')
    args = parser.parse_args()

    """get sentence-level gleu scores"""
    sys.stderr.write('Running GLEU...\n')
    gleu_calculator = GLEU(args.n)
    gleu_calculator.load_sources(args.src)
    gleu_calculator.load_references(args.ref)
    for hpath in args.hyp:
        print(hpath)
        print([g for g in gleu_calculator.run_iterations(num_iterations=args.iter,
                                                         source=args.src,
                                                         hypothesis=hpath,
                                                         per_sent=args.sent)])
