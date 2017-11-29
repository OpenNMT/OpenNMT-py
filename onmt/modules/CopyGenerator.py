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
    def __init__(self, opt, src_dict, tgt_dict, pointer_gen):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(opt.rnn_size, len(tgt_dict))

        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.pointer_gen = pointer_gen
        if self.pointer_gen:
            self.linear_hidden = nn.Linear(opt.rnn_size, 1)
            self.linear_decoder_state = nn.Linear(opt.rnn_size, 1)
            self.linear_decoder_input = nn.Linear(opt.tgt_word_vec_size, 1)
        else:
            self.linear_copy = nn.Linear(opt.rnn_size, 1)

    def forward(self, hidden, attn, src_map,
                rnn_output, src_emb):
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
        if self.pointer_gen:
            """
            p_gen = sigm(w1*hidden + w2*decoder_state + w3*decoder_input)
            
            hidden = post-attention hidden_state
            decoder_state = pre-attention hidden_state 
            """
            copy = F.sigmoid(
                self.linear_hidden(hidden)
                + self.linear_decoder_state(rnn_output)
                + self.linear_decoder_input(src_emb)
            )
        else:
            copy = F.sigmoid(self.linear_copy(hidden))

        # Probability of not copying: p_{word}(w) * (1 - p(z))
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
                 force_copy, pointer_gen, eps=1e-20):
        super(CopyGeneratorLossCompute, self).__init__(generator, tgt_vocab)

        self.dataset = dataset
        self.force_copy = force_copy
        self.criterion = CopyGeneratorCriterion(len(tgt_vocab), force_copy,
                                                self.padding_idx)
        self.pointer_gen = pointer_gen

    def make_shard_state(self, batch, output, range_, attns,
                         rnn_outputs, src_emb):
        """ See base class for args description. """
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn or -pointer_gen "
                                 "you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        if self.pointer_gen:
            generator_attn = attns.get("std")
        else:
            generator_attn = attns.get("copy")
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "generator_attn": generator_attn,
            "align": batch.alignment[range_[0] + 1: range_[1]],
            "rnn_outputs": rnn_outputs,
            "src_emb": src_emb,
        }

    def compute_loss(self, batch, output, target, generator_attn, align,
                     rnn_outputs, src_emb):
        """
        Compute the loss. The args must match self.make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            generator_attn: attention for generator
            align: the align info.
        """
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(self.bottle(output),
                                self.bottle(generator_attn),
                                batch.src_map,
                                self.bottle(rnn_outputs),
                                self.bottle(src_emb),
                                )

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
