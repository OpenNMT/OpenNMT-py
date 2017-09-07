from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt
import onmt.modules
from onmt.Loss import RLStatistics


def sample_gumbel(type_template, size, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = type_template.data.new(*size).uniform_()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + Variable(sample_gumbel(logits, logits.size()))
    return torch.nn.functional.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    y_max_score, y_max_ind = y.max(1)
    if hard:
        y_hard = y.eq(y_max_score.unsqueeze(1)).type_as(y)
        y = (y_hard - y).detach() + y
    return y, y_max_ind


class MRTAgent(nn.Module):

    def __init__(self, model, generator, scorer, opt):
        super(MRTAgent, self).__init__()
        self.model = model
        self.generator = generator
        self.scorer = scorer
        self.use_gpu = len(opt.gpus) > 0
        self.multi_gpu = len(opt.gpus) > 1
        # self.max_len = opt.max_sent_length
        self.max_len = 50
        self.max_len_ratio = 1.5
        # self.n_sample = opt.rl_sample_size
        self.n_sample = 20
        self.alpha = 5e-3
        self.temperature = 1.0
        self.gumbel = False

    def forward(self, src, tgt, lengths, largest_len=None):
        tt = torch.cuda if self.use_gpu else torch

        # 1. Encode
        encStates, contexts, embs = self.model.encoder(
            src, lengths=lengths, largest_len=largest_len
        )
        seqL, batchSize, rnnSize = contexts.size()

        encStates = self.model.init_decoder_state(contexts, encStates)

        padMask = src[:, :, 0].data.eq(onmt.Constants.PAD).t()

        def mask(padMask):
            self.model.decoder.attn.applyMask(padMask)

        totalLoss = Variable(contexts.data.new([[0]]))  # -firstMoment
        firstMoment = Variable(contexts.data.new([[0]]))
        secondMoment = Variable(contexts.data.new([[0]]))
        numCorrect = Variable(contexts.data.new([[0]]))
        numWords = Variable(contexts.data.new([[0]]))

        # 2. For each encoded sample in a batch, decode
        for encState, context, emb, pad, src_t, targ_t in zip(encStates.split(1, 1, 0),
                                                              contexts.split(1, 1),
                                                              embs.detach().split(1, 1),
                                                              padMask.split(1, 0),
                                                              src.split(1, 1),
                                                              tgt.split(1, 1)):

            # a. Prepare anything for every sequence to sample
            encState.expandAsBatch_(self.n_sample)
            contexts_ = context.expand(seqL, self.n_sample, rnnSize)
            embs_ = emb.expand(seqL, self.n_sample, rnnSize)
            padMask = pad.expand(self.n_sample, seqL)

            targ_t = targ_t.expand(targ_t.size(0), self.n_sample).t()
            src_t = src_t.expand(src_t.size(0), self.n_sample, src_t.size(2))

            max_len = int(self.max_len_ratio * lengths.data.max())+2

            decStates = encState

            resultSeqs = [[onmt.Constants.BOS] for _ in range(self.n_sample)]
            accumScores = [Variable(contexts.data.new([0])) for _ in range(self.n_sample)]
            curSeqs = resultSeqs
            curAccumScores = accumScores

            input = Variable(targ_t.data.new(1, self.n_sample).fill_(onmt.Constants.BOS))
            n_old_active = self.n_sample
            mask(padMask.unsqueeze(0))

            # b. Begin Sampling
            for i in range(1, max_len):
                decOut, decStates, attn = self.model.decoder(input,
                                                             src_t,
                                                             contexts_,
                                                             decStates,
                                                             embs_)

                decOut = decOut.squeeze(0)
                out = self.generator.forward(decOut)
                if self.gumbel:
                    onehots, pred_t = gumbel_softmax(out, self.temperature)
                    pred_t = pred_t.unsqueeze(1)
                    onehots = onehots.unsqueeze(0)
                else:
                    pred_t = out.exp().multinomial(1).detach()

                active = Variable(pred_t.data.ne(onmt.Constants.EOS).squeeze()
                                                                    .nonzero()
                                                                    .squeeze())

                score_t = out.gather(1, pred_t).squeeze(-1)
                tokens = pred_t.squeeze().data

                # Update scores and sequences
                for i in range(len(curAccumScores)):
                    curAccumScores[i] += score_t[i]
                    curSeqs[i].append(tokens[i])

                # If none is active, then stop
                if len(active.size()) == 0 or active.size(0) == 0:
                    break

                if active.size(0) == n_old_active:
                    input = onehots if self.gumbel else pred_t.t()
                    continue

                n_old_active = active.size(0)

                newAccumScores = []
                newSeqs = []
                for b in list(active.data):
                    newAccumScores.append(curAccumScores[b])
                    newSeqs.append(curSeqs[b])
                curAccumScores = newAccumScores
                curSeqs = newSeqs

                input = onehots.index_select(1, active) \
                    if self.gumbel else pred_t.t().index_select(1, active)
                src_t = src_t.index_select(1, active)
                contexts_ = contexts_.index_select(1, active)
                embs_ = embs_.index_select(1, active)
                padMask = padMask.index_select(0, active.data)
                mask(padMask.unsqueeze(0))
                decStates.activeUpdate_(active)

            accumScores = torch.cat(accumScores, 0)

            bleu = self.scorer.score(resultSeqs, targ_t)
            bleu = Variable(contexts.data.new(bleu))
            prob = nn.functional.softmax(accumScores*self.alpha)
            # loss = - bleu * prob

            loss = - accumScores * bleu

            # import os
            # from tensorboardX import SummaryWriter
            # import random
            # import time
            # time.sleep(random.randint(1, 10))
            # if not os.path.exists('runs/graph_view'):
            #     writer = SummaryWriter(log_dir='runs/graph_view')
            #     writer.add_graph(self.model.encoder, contexts)
            #     writer.close()

            # import pdb; pdb.set_trace()

            # totalLoss += loss.sum()
            totalLoss += loss.mean()

            # For monitoring purposes
            firstMoment += (bleu * prob.detach()).sum()
            secondMoment += (bleu.pow(2) * prob.detach()).sum()

            numWords += targ_t.data.ne(onmt.Constants.PAD).sum()
            # print sequences.size(), targ_t.size()
            # if sequences.size(1) >= targ_t.size(1):
            #     sequences = sequences[:, :targ_t.size(1)]
            #     non_padding = targ_t.ne(onmt.Constants.PAD).data
            #     numWords += non_padding.sum()
            #     numCorrect += sequences.data.eq(targ_t.data) \
            #                            .masked_select(non_padding) \
            #                            .sum()
            # else:
            #     numWords += targ_t.ne(onmt.Constants.PAD).data.sum()
            #     targ_t = targ_t[:, :sequences.size(1)]
            #     non_padding = targ_t.ne(onmt.Constants.PAD).data
            #     numCorrect += sequences.data.eq(targ_t.data) \
            #                            .masked_select(non_padding) \
            #                            .sum()

        self.model.decoder.attn.removeMask()
        return totalLoss, firstMoment, secondMoment, numWords


class MRT(object):

    def __init__(self, model, generator, scorer, opt):
        self.step = MRTAgent(model, generator, scorer, opt)
        self.model = model
        self.generator = generator
        self.gpus = opt.gpus
        self.mini_size = 2

    def policy_grad(self, batch):
        rl_statistics = RLStatistics()
        for b in batch.xsplit(self.mini_size):
            # import pdb; pdb.set_trace()
            total_loss, first_moment, second_moment, n_words = \
                                    self.step(b.src, b.tgt, b.lengths,
                                              largest_len=b.lengths.data.max())

            sum_loss = total_loss.sum()
            sum_loss.div(batch.batchSize).backward()
            rl_statistics.update(RLStatistics(first_moment.data.sum(),
                                              second_moment.data.sum(),
                                              n_words.data.sum(),
                                              0,
                                              self.mini_size))
        # for p in self.model.parameters():
        #     print p.grad.data.sum()
        # print 'damn'
        # for p in self.step.module.model.parameters():
        #     print p.grad.data.sum()
        # print 'damn1'

        return rl_statistics
