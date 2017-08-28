from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt
import onmt.modules
from onmt.Loss import RLStatistics


class MRTStep(nn.Module):

    def __init__(self, model, generator, scorer, opt):
        super(MRTStep, self).__init__()
        self.model = model
        self.generator = generator
        self.scorer = scorer
        self.use_gpu = len(opt.gpus) > 0
        self.multi_gpu = len(opt.gpus) > 1
        # self.max_len = opt.max_sent_length
        self.max_len = 50
        self.max_len_ratio = 1.5
        # self.n_sample = opt.rl_sample_size
        self.n_sample = 100
        self.alpha = 5e-3

    def forward(self, src, tgt, lengths, largest_len=None):
        tt = torch.cuda if self.use_gpu else torch
        # targ_t = tgt.t().contiguous()

        encStates, contexts = self.model.encoder(
            src, lengths=lengths, largest_len=largest_len
        )
        seqL, batchSize, rnnSize = contexts.size()

        # print 'context', contexts.size(), ' src', src.size()

        encStates = self.model.init_decoder_state(contexts, encStates)

        padMask = src[:, :, 0].data.eq(onmt.Constants.PAD)

        def mask(padMask):
            self.model.decoder.attn.applyMask(padMask)

        totalLoss = Variable(tt.FloatTensor([[0]]))  # -firstMoment
        secondMoment = Variable(tt.FloatTensor([[0]]))
        numCorrect = Variable(tt.FloatTensor([[0]]))
        numWords = Variable(tt.FloatTensor([[0]]))

        for encState, context, pad, src_t, targ_t in zip(encStates.split(1, 1, 0),
                                                         contexts.split(1, 1),
                                                         padMask.split(1, 1),
                                                         src.split(1, 1),
                                                         tgt.split(1, 1)):
            # import pdb; pdb.set_trace()

            encState.expandAsBatch_(self.n_sample)

            contexts = context.expand(seqL, self.n_sample, rnnSize)
            padMask = pad.expand(seqL, self.n_sample).t()

            targ_t = targ_t.expand(targ_t.size(0), self.n_sample).t()
            src_t = src_t.expand(src_t.size(0), self.n_sample, src_t.size(2))

            max_len = int(self.max_len_ratio * lengths.data.max())+2

            accumScores = Variable(contexts.data.new(self.n_sample).zero_())
            decStates = encState
            sequences = []
            dead = Variable(tt.ByteTensor(self.n_sample).zero_())
            input = Variable(targ_t.data.new(1, self.n_sample).fill_(onmt.Constants.BOS))
            sequences.append(input.squeeze(0))

            for i in range(1, max_len):
                decOut, decStates, attn = self.model.decoder(input,
                                                             src_t,
                                                             contexts,
                                                             decStates)
                decOut = decOut.squeeze(0)
                out = self.generator.forward(decOut)
                pred_t = out.exp().multinomial(1).detach()

                score_t = out.gather(1, pred_t).squeeze(-1) * (1-dead.float())
                accumScores += score_t

                input = pred_t.t().contiguous()
                sequences.append(input.squeeze(0))

                newDead = pred_t.eq(onmt.Constants.EOS).squeeze()
                dead = dead.max(newDead)
                if dead.sum().data[0] == dead.size(0):
                    break

            sequences = torch.stack(sequences, 1)
            bleu = self.scorer.score(sequences, targ_t)
            bleu = Variable(tt.FloatTensor(bleu))
            prob = nn.functional.softmax(accumScores*self.alpha)
            loss = - bleu * prob

            totalLoss += loss.sum()

            # For monitoring purposes
            secondMoment += (bleu.pow(2) * prob.detach()).sum()

            # print sequences.size(), targ_t.size()
            if sequences.size(1) >= targ_t.size(1):
                sequences = sequences[:, :targ_t.size(1)]
                non_padding = targ_t.ne(onmt.Constants.PAD).data
                numWords += non_padding.sum()
                numCorrect += sequences.data.eq(targ_t.data) \
                                       .masked_select(non_padding) \
                                       .sum()
            else:
                numWords += targ_t.ne(onmt.Constants.PAD).data.sum()
                targ_t = targ_t[:, :sequences.size(1)]
                non_padding = targ_t.ne(onmt.Constants.PAD).data
                numCorrect += sequences.data.eq(targ_t.data) \
                                       .masked_select(non_padding) \
                                       .sum()

        self.model.decoder.attn.removeMask()
        return totalLoss, secondMoment, numWords, numCorrect


class MRT(object):

    def __init__(self, model, generator, scorer, opt):
        self.step = MRTStep(model, generator, scorer, opt)
        self.model = model
        self.mini_size = 2

    def policy_grad(self, batch):
        rl_statistics = RLStatistics()
        for b in batch.xsplit(self.mini_size):
            # import pdb; pdb.set_trace()
            total_loss, second_moment, n_words, n_correct = \
                                    self.step(b.src, b.tgt, b.lengths,
                                              largest_len=b.lengths.data.max())

            sum_loss = total_loss.sum()
            sum_loss.div(batch.batchSize).backward()
            rl_statistics.update(RLStatistics(-sum_loss.data[0],
                                              second_moment.data.sum(),
                                              n_words.data.sum(),
                                              n_correct.data.sum(),
                                              self.mini_size))
        # for p in self.model.parameters():
        #     print p.grad.data.sum()
        # print 'damn'
        # for p in self.step.module.model.parameters():
        #     print p.grad.data.sum()
        # print 'damn1'

        return rl_statistics
