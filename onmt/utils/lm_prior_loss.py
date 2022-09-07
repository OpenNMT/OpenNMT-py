# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

# Special tokens for LM mode
LM_BOS_idx = 0
UNK_idx = 0
PAD_idx = 1
LM_EOS_idx = 2
# => vocab start at idx 3
# Special tokens for MT mode
UNK_idx = 0
PAD_idx = 1
MT_BOS_idx = 2
MT_EOS_idx = 3
# => vocab start at idx 4

# in MT mode we have the following
# SRC: no BOS, no EOS
# TGT: BOS ... EOS
# in LM mode we have the following
# SRC: BOS ... no BOS
# TGT: no BOS ... EOS


def lm_prior_loss(model, lm_prior_model, outputs, tgt, lm_prior_tau):
    """Compute the loss between MT output and LM output"""
    # https://github.com/cbaziotis/lm-prior-for-nmt/blob/master
    # /fairseq_extension/user/lm_prior/lm_prior.py#L131-L133

    scores = model.generator(outputs.view(-1, outputs.size(2)) / lm_prior_tau)
    # <--- important to make this independant
    lm_src = tgt.detach().clone()
    # using here the onmt-py model but very slow for a good
    # model - ct2 not usable at the moment
    # matching ids because LM and MT have different special tokens
    # for regular vocab
    lm_src = torch.where((lm_src == MT_BOS_idx) |
                         (lm_src == UNK_idx), LM_BOS_idx,
                         torch.where(lm_src == PAD_idx, PAD_idx, lm_src - 1))
    lm_tgt = lm_src[1:, :, :]    # we remove the BOS
    lm_src = lm_src[:-1, :, :]   # we remove the EOS
    # lm_src is [max_length, batch_size, 1]
    lm_src_lengths = lm_src[:, :, 0].ne(PAD_idx).sum(0).int()
    lm_outs, _ = lm_prior_model(lm_src, lm_tgt,
                                lm_src_lengths, with_align=False)
    # lm_outs is [max_length, batch_size, rnn size]
    lm_scores = lm_prior_model.generator(lm_outs.view(-1, lm_outs.size(2))
                                         / lm_prior_tau)
    # lm_scores is [max_length x batch_size, vocab_size]

    # the we need to match back ids to align with TM vocabs ids
    add_vocab_lm = torch.zeros(lm_scores.size(0), 1).\
        to(lm_scores.get_device())
    add_vocab_lm[:, 0] = -100   # add token UNK with low prob
    lm_scores = torch.cat((add_vocab_lm, lm_scores), dim=1)
    bos_scores = lm_scores[:, 1]
    pad_scores = lm_scores[:, 2]
    eos_scores = lm_scores[:, 3] - 20
    lm_scores[:, 1] = pad_scores
    lm_scores[:, 2] = bos_scores
    lm_scores[:, 3] = eos_scores
    lm_loss = F.kl_div(scores, lm_scores, reduction='batchmean',
                       log_target=True) * lm_prior_tau * lm_prior_tau
    return lm_loss
