"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np
import onmt.modules
from onmt.modules import aeq


def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_k = seq_k.size()
    mb_size_, len_q = seq_q.size()
    aeq(mb_size, mb_size_)
    # bx1xsk
    pad_attn_mask = seq_k.data.eq(onmt.Constants.PAD).unsqueeze(1) \
        .expand(mb_size, len_q, len_k)
    return pad_attn_mask


def get_attn_subsequent_mask(size):
    ''' Get an attention mask to avoid using the subsequent info.'''
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = onmt.modules.BottleLinear(d_hid, d_inner_hid)
        self.w_2 = onmt.modules.BottleLinear(d_inner_hid, d_hid)
        self.layer_norm = onmt.modules.BottleLayerNorm(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.dropout(self.w_2(self.relu(self.w_1(x))))
        return self.layer_norm(output + residual)


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, opt, n_head=8, d_inner=2048):
        super(TransformerEncoder, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            n_head, hidden_size, p=opt.dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size,
                                                    d_inner,
                                                    opt.dropout)

    def forward(self, input, words):
        # CHECKS
        n_batch, s_len, _ = input.size()
        n_batch_, s_len_ = words.size()
        aeq(n_batch, n_batch_)
        aeq(s_len, s_len_)
        # END CHECKS

        mask = get_attn_padding_mask(words, words)
        mid, _ = self.self_attn(input, input, input, mask=mask)
        out = self.feed_forward(mid)
        return out


class TransformerDecoder(nn.Module):
    """
    The Transformer Decoder from AIAYN
    """
    def __init__(self, hidden_size, opt, n_head=8, d_inner=2048):
        super(TransformerDecoder, self).__init__()
        self.self_attn = onmt.modules.MultiHeadedAttention(n_head, hidden_size,
                                                           p=opt.dropout)
        self.context_attn = onmt.modules.MultiHeadedAttention(n_head,
                                                              hidden_size,
                                                              p=opt.dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size,
                                                    d_inner,
                                                    opt.dropout)
        self.dropout = opt.dropout
        self.mask = get_attn_subsequent_mask(5000).cuda()

    def forward(self, input, context, src_words, tgt_words):
        # CHECKS
        n_batch, t_len, _ = input.size()
        n_batch_, s_len, _ = context.size()
        n_batch__, s_len_ = src_words.size()
        n_batch___, t_len_ = tgt_words.size()
        aeq(n_batch, n_batch_, n_batch__, n_batch___)
        aeq(s_len, s_len_)
        aeq(t_len, t_len_)
        # END CHECKS

        attn_mask = get_attn_padding_mask(tgt_words, tgt_words)
        dec_mask = torch.gt(attn_mask + self.mask[:, :attn_mask.size(1),
                                                  :attn_mask.size(1)]
                            .expand_as(attn_mask), 0)

        pad_mask = get_attn_padding_mask(tgt_words, src_words)
        query, attn = self.self_attn(input, input, input, mask=dec_mask)
        mid, attn = self.context_attn(context, context, query, mask=pad_mask)
        output = self.feed_forward(mid)

        return output, attn
