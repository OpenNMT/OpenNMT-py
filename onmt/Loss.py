import onmt
import onmt.Constants
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import sys
import math


class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

class FocalLoss(_Loss):
    r"""The focal loss. It is useful to train a classification
    problem with n classes
    """

    def __init__(self, size_average=True, gamma=2.0, alpha=0.5):
        super(FocalLoss, self).__init__(size_average)
        self.gamme = gamme
        self.alpha = alpha

    def forward(self, input, target):

        def _assert_no_grad(variable):
            assert not variable.requires_grad, \
                "nn criterions don't compute the gradient w.r.t. targets - please " \
                "mark these variables as volatile or not requiring gradients"

        _assert_no_grad(target)
        loss = input.gather(1, target)
        loss = - self.alpha * loss * (1-loss.exp()).pow(self.gamma)
        return loss.mean() if self.size_average else loss.sum()

def NMTCriterion(vocabSize, opt):
    """
    Construct the standard NMT Criterion
    """
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


def shardVariables(variables, batches, eval):
    """
    Split a dict of variables up into sharded dummy
    variables.
    """
    dummies = {}
    n_shards = ((list(variables.values())[0].size(0) - 1) // batches) + 1
    shards = [{} for _ in range(n_shards)]
    for k in variables:
        if isinstance(variables[k], Variable) and variables[k].requires_grad:
            dummies[k] = Variable(variables[k].data, requires_grad=(not eval),
                                  volatile=eval)
        else:
            dummies[k] = variables[k]
        splits = torch.split(dummies[k], batches)
        for i, v in enumerate(splits):
            shards[i][k] = v
    return shards, dummies


def collectGrads(variables, dummy):
    """Given a set of variables, find the ones with gradients"""
    inputs = []
    grads = []
    for k in dummy:
        if isinstance(variables[k], Variable) and (dummy[k].grad is not None):
            inputs.append(variables[k])
            grads.append(dummy[k].grad.data)
    return inputs, grads


class BasicStatistics(object):
    """
    Template of Statistics
    """
    def __init__(self, n_words=0, n_correct=0, n_sents=0):
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_sents = n_sents
        self.n_src_words = 0
        self.start_time = time.time()

    def update_basic(self, stat):
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.n_sents += stat.n_sents

    def log_basic(self, prefix, experiment, optim):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", optim.lr)

    def accuracy(self):
        return 100 * (self.n_correct / float(self.n_words))

    def elapsed_time(self):
        return time.time() - self.start_time


class Statistics(BasicStatistics):
    """
    Training loss function statistics.
    """
    def __init__(self, loss=0, bleu=0, *args, **kargs):
        super(Statistics, self).__init__(*args, **kargs)
        self.loss = loss
        self.total_bleu = bleu

    def update(self, stat):
        self.update_basic(stat)
        self.loss += stat.loss
        self.total_bleu += stat.total_bleu

    def bleu(self):
        return 100 * (self.total_bleu / float(self.n_sents))

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def output(self, epoch, batch, n_batches, start):
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %5.2f; ppl: %5.2f; bleu: %4.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %5.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.bleu(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, optim):
        self.log_basic(prefix, experiment, optim)
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_bleu", self.bleu())


class RLStatistics(BasicStatistics):
    """
    Statistics for reinforce learning
    """
    def __init__(self, bleu=0, bleu_2=0, *args, **kargs):
        super(RLStatistics, self).__init__(*args, **kargs)
        self.bleu = bleu
        self.bleu_2 = bleu_2

    def update(self, stat):
        self.update_basic(stat)
        self.bleu += stat.bleu
        self.bleu_2 += stat.bleu_2

    def bleu_mean(self):
        return 100 * (self.bleu / float(self.n_sents))

    def bleu_std(self):
        return math.sqrt(self.bleu_2 * 10000 / float(self.n_sents)
                         - self.bleu_mean() ** 2)

    def output(self, epoch, batch, n_batches, start):
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %5.2f; bleu: %4.2f; std: %4.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %5.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.bleu_mean(),
               self.bleu_std(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, optim):
        self.log_basic(prefix, experiment, optim)
        experiment.add_scalar_value(prefix + "_bleu", self.bleu_mean())
        experiment.add_scalar_value(prefix + "_bleu_std", self.bleu_std())


class BleuScore:

    def getRefDict(self, words, ngram):
        '''
            Get the count of n-grams in the reference

            :type words: list
            :param words: indexed sentence

            :type ngram: int
            :param ngram: maximum length of counted n-grams
        '''
        lens = len(words)
        now_ref_dict = {}
        for n in range(1, ngram+1):
            for start_ in range(lens - n + 1):
                gram = ' '.join([str(p) for p in words[start_: start_ + n]])
                # print n, gram
                now_ref_dict[gram] = now_ref_dict.get(gram, 0) + 1
        return now_ref_dict, lens

    def calBleu(self, x, ref_dict, lens, ngram):
        '''
            Calculate BLEU score with single reference

            :type x: list
            :param x: the indexed hypothesis sentence

            :type ref_dict: dict
            :param ref_dict: the n-gram count generated by getRefDict()

            :type lens: int
            :pais not None ram lens: the length of the reference

            :type ngram: int
            :param ngram: maximum length of counted n-grams
        '''
        length_trans = len(x)
        words = x
        closet_length = lens
        sent_dict = {}
        for n in range(1, ngram + 1):
            for start in range(length_trans - n + 1):
                gram = ' '.join([str(p) for p in words[start: start + n]])
                sent_dict[gram] = sent_dict.get(gram, 0) + 1
        correct_gram = [0] * ngram
        # print ref_dict
        for gram in sent_dict:
            if gram in ref_dict:
                n = len(gram.split(' '))
                correct_gram[n - 1] += min(ref_dict[gram], sent_dict[gram])
        bleu = [0.] * ngram
        smooth = 0
        for j in range(ngram):
            if correct_gram[j] == 0:
                smooth = 1
        for j in range(ngram):
            if length_trans > j:
                bleu[j] = 1. * (correct_gram[j] + smooth) / \
                                (length_trans - j + smooth)
            else:
                bleu[j] = 1
        brev_penalty = 1
        if length_trans < closet_length:
            brev_penalty = math.exp(1 - closet_length * 1. / length_trans)
        logsum = 0
        for j in range(ngram):
            # print j, bleu[j]
            logsum += math.log(bleu[j])
        now_bleu = brev_penalty * math.exp(logsum / ngram)
        return now_bleu

    def score(self, pred_t, targ_t, ngram=4):
        bleus = []
        # print pred_t.size()
        for i in range(targ_t.size(0)):
            targ_seq = list(targ_t.data[i])
            if onmt.Constants.EOS in targ_seq:
                targ_seq = targ_seq[:targ_seq.index(onmt.Constants.EOS)+1]
            pred_seq = list(pred_t.data[i])
            if onmt.Constants.EOS in pred_seq:
                pred_seq = pred_seq[:pred_seq.index(onmt.Constants.EOS)+1]
            ref_dict, targ_len = self.getRefDict(targ_seq, ngram)
            bleu = self.calBleu(pred_seq, ref_dict, targ_len, ngram)
            bleus.append(bleu)

        return bleus


class MemoryEfficientLoss:
    """
    Class for best batchin the loss for NMT.
    """
    def __init__(self, opt, generator, crit,
                 copy_loss=False,
                 coverage_loss=False,
                 eval=False,
                 rl=False):
        """
        Args:
            generator (Function): ( any x rnn_size ) -> ( any x tgt_vocab )
            crit (Criterion): ( any x tgt_vocab )
            eval (bool): train or eval
        """
        self.generator = generator
        self.crit = crit
        self.eval = eval
        self.max_batches = opt.max_generator_batches
        self.copy_loss = copy_loss
        self.lambda_coverage = opt.lambda_coverage
        self.coverage_loss = coverage_loss
        self.cuda = len(opt.gpus) > 0
        self.rl = rl
        self.bleu_scorer = BleuScore()

    def score(self, loss_t, scores_t, targ_t, bleu=0):
        pred_t = scores_t.data.max(1)[1]
        non_padding = targ_t.ne(onmt.Constants.PAD).data
        num_correct_t = pred_t.eq(targ_t.data) \
                              .masked_select(non_padding) \
                              .sum()
        return Statistics(loss_t.data[0], bleu, non_padding.sum(),
                          num_correct_t, n_sents=targ_t.size(0))

    def compute_std_loss(self, out_t, targ_t):
        scores_t = self.generator(out_t)
        loss_t = self.crit(scores_t, targ_t.view(-1))
        return loss_t, scores_t

    def compute_copy_loss(self, out_t, targ_t, attn_t, align_t):
        scores_t, c_attn_t = self.generator(out_t, attn_t)
        loss_t = self.crit(scores_t, c_attn_t, targ_t, align_t)
        return loss_t, scores_t

    def loss(self, batch, outputs, attns, bleu=False):
        """
        Args:
            batch (Batch): Data object
            outputs (FloatTensor): tgt_len x batch x rnn_size
            attns (dictionary): Dictionary of attention objects
        Returns:
            stats (Statistics): Statistics about loss
            inputs: list of variables with grads
            grads: list of grads corresponding to inputs
        """
        stats = Statistics()

        original = {"out_t": outputs.transpose(0, 1).contiguous(),
                    "targ_t": batch.tgt[1:].transpose(0, 1).contiguous()}

        if self.coverage_loss:
            original["coverage_t"] = attns["coverage"].transpose(0, 1).contiguous()

        if self.copy_loss:
            original["attn_t"] = attns["copy"].transpose(0, 1).contiguous()
            original["align_t"] = batch.alignment[1:].transpose(0, 1).contiguous()

        shards, dummies = shardVariables(original, self.max_batches, self.eval)

        def bottle(v):
            return v.view(-1, v.size(2))
        for s in shards:
            if not self.copy_loss:
                loss_t, scores_t = self.compute_std_loss(bottle(s["out_t"]),
                                                         s["targ_t"])
            else:
                loss_t, scores_t = self.compute_copy_loss(
                    bottle(s["out_t"]), s["targ_t"],
                    bottle(s["attn_t"]), bottle(s["align_t"]))

            if self.coverage_loss:
                loss_t += self.lambda_coverage * torch.min(s["coverage"],
                                                           s["attn"]).sum()

            b, l, _ = s["out_t"].size()
            scores_rec = scores_t.view(b, l, -1)
            bleu = self.bleu_scorer.score(scores_rec.max(-1)[1].squeeze(-1), s["targ_t"])

            stats.update(self.score(loss_t, scores_t, s["targ_t"], sum(bleu)))
            # stats.update(self.score(loss_t, scores_t, s["targ_t"], 0))
            if not self.eval:
                loss_t.div(batch.batchSize).backward()

        # Return the gradients
        inputs, grads = collectGrads(original, dummies)
        return stats, inputs, grads
