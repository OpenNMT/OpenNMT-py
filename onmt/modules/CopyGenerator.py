import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda

import onmt
from onmt.Utils import aeq


class CopyGenerator(nn.Module):
    """
    Generator module that additionally considers copying
    words directly from the source.
    """
    def __init__(self, opt, src_dict, tgt_dict):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(opt.rnn_size, len(tgt_dict))
        self.linear_copy = nn.Linear(opt.rnn_size, 1)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    def forward(self, hidden, attn, src_map):
        """
        Computes p(w) = p(z=1) p_{copy}(w|z=0)  +  p(z=0) * p_{softmax}(w|z=0)
        """
        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.tgt_dict.stoi[onmt.IO.PAD_WORD]] = -float('inf')
        prob = F.softmax(logits)

        # Probability of copying p(z=1) batch.
        copy = F.sigmoid(self.linear_copy(hidden))

        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob,  1 - copy.expand_as(prob))
        mul_attn = torch.mul(attn, copy.expand_as(attn))
        copy_prob = torch.bmm(mul_attn.view(-1, batch, slen)
                              .transpose(0, 1),
                              src_map.transpose(0, 1)).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)


class CopyGeneratorCriterion(object):
    def __init__(self, vocab_size, force_copy, pad, eps=1e-20):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = vocab_size
        self.pad = pad

    def __call__(self, scores, align, target):
        align = align.view(-1)

        # Copy prob.
        out = scores.gather(1, align.view(-1, 1) + self.offset) \
                    .view(-1).mul(align.ne(0).float())
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)

        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            out = out + self.eps + tmp.mul(target.ne(0).float()) + \
                  tmp.mul(align.eq(0).float()).mul(target.eq(0).float())
        else:
            # Forced copy.
            out = out + self.eps + tmp.mul(align.eq(0).float())

        # Drop padding.
        loss = -out.log().mul(target.ne(self.pad).float()).sum()
        return loss


class CopyGeneratorLossCompute(onmt.Loss.LossComputeBase):
    """
    Copy Generator Loss Computation.
    """
    def __init__(self, generator, tgt_vocab, dataset,
                 force_copy, eps=1e-20):
        super(CopyGeneratorLossCompute, self).__init__(generator, tgt_vocab)

        self.dataset = dataset
        self.copy_attn = True
        self.force_copy = force_copy
        self.criterion = CopyGeneratorCriterion(len(tgt_vocab), force_copy,
                                                self.padding_idx)

    def compute_loss(self, batch, output, target, copy_attn, align):
        """
        Compute the loss. The args must match Loss.make_gen_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(self.bottle(output),
                                self.bottle(copy_attn),
                                batch.src_map)

        loss = self.criterion(scores, align, target)

        scores_data = scores.data.clone()
        scores_data = self.dataset.collapse_copy_scores(
                self.unbottle(scores_data, batch.batch_size),
                batch, self.tgt_vocab)
        scores_data = self.bottle(scores_data)

        # Correct target is copy when only option.
        # TODO: replace for loop with masking or boolean indexing
        target_data = target.data.clone()
        for i in range(target_data.size(0)):
            if target_data[i] == 0 and align.data[i] != 0:
                target_data[i] = align.data[i] + len(self.tgt_vocab)

        # Coverage loss term.
        loss_data = loss.data.clone()

        stats = self.stats(loss_data, scores_data, target_data)

        return loss, stats
