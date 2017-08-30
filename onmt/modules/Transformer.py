"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np
import onmt.modules
from onmt.modules import aeq


MAX_SIZE = 5000


def get_attn_padding_mask(seq_q, seq_k, padding_idx):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_k = seq_k.size()
    mb_size_, len_q = seq_q.size()
    aeq(mb_size, mb_size_)
    # bx1xsk
    pad_attn_mask = seq_k.data.eq(padding_idx).unsqueeze(1) \
        .expand(mb_size, len_q, len_k)
    return pad_attn_mask


def get_attn_subsequent_mask(size):
    ''' Get an attention mask to avoid using the subsequent info.'''
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network."""
    def __init__(self, size, hidden_size, dropout=0.1):
        """
        Args:
            size(int): the size of input for the first-layer of the FFN.
            hidden_size(int): the hidden layer size of the second-layer
                              of the FNN.
            droput(float): dropout probability(0-1.0).
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = onmt.modules.BottleLinear(size, hidden_size)
        self.w_2 = onmt.modules.BottleLinear(hidden_size, size)
        self.layer_norm = onmt.modules.BottleLayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.dropout(self.w_2(self.relu(self.w_1(x))))
        return self.layer_norm(output + residual)


class TransformerEncoder(nn.Module):
    """
    The Transformer Decoder from "Attention is All You Need".
    """
    def __init__(self, size, dropout, padding_idx,
                 head_count=8, hidden_size=2048):
        """
        Args:
            size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
            droput(float): dropout probability(0-1.0).
            padding_idx(id): the padding character idx in the Vocabulary.
            head_count(int): the number of head for MultiHeadedAttention.
            hidden_size(int): the second-layer of the PositionwiseFeedForward.
        """
        super(TransformerEncoder, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            head_count, size, p=dropout)
        self.feed_forward = PositionwiseFeedForward(size,
                                                    hidden_size,
                                                    dropout)
        self.padding_idx = padding_idx

    def forward(self, input, words):
        # CHECKS
        n_batch, s_len, _ = input.size()
        n_batch_, s_len_ = words.size()
        aeq(n_batch, n_batch_)
        aeq(s_len, s_len_)
        # END CHECKS

        mask = get_attn_padding_mask(words, words, self.padding_idx)
        mid, _ = self.self_attn(input, input, input, mask=mask)
        out = self.feed_forward(mid)
        return out


class TransformerDecoder(nn.Module):
    """
    The Transformer Decoder from AIAYN
    """
    def __init__(self, size, dropout, padding_idx,
                 head_count=8, hidden_size=2048):
        """
        Args:
            size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
            droput(float): dropout probability(0-1.0).
            padding_idx(id): the padding character idx in the Vocabulary.
            head_count(int): the number of head for MultiHeadedAttention.
            hidden_size(int): the second-layer of the PositionwiseFeedForward.
        """
        super(TransformerDecoder, self).__init__()
        self.self_attn = onmt.modules.MultiHeadedAttention(
                head_count, size, p=dropout)
        self.context_attn = onmt.modules.MultiHeadedAttention(
                head_count, size, p=dropout)
        self.feed_forward = PositionwiseFeedForward(size,
                                                    hidden_size,
                                                    dropout)
        self.dropout = dropout
        self.mask = get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoder, so
        # it gets TransformerDecoder's cuda behavior automatically.
        self.register_buffer('mask', self.mask)
        self.padding_idx = padding_idx

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

        attn_mask = get_attn_padding_mask(tgt_words, tgt_words,
                                          self.padding_idx)
        dec_mask = torch.gt(attn_mask + self.mask[:, :attn_mask.size(1),
                                                  :attn_mask.size(1)]
                            .expand_as(attn_mask), 0)

        pad_mask = get_attn_padding_mask(tgt_words, src_words,
                                         self.padding_idx)
        query, attn = self.self_attn(input, input, input, mask=dec_mask)
        mid, attn = self.context_attn(context, context, query, mask=pad_mask)
        output = self.feed_forward(mid)

        # CHECKS
        n_batch_, t_len_, _ = output.size()
        aeq(t_len, t_len_)
        aeq(n_batch, n_batch_)

        n_batch_, t_len_, s_len_ = attn.size()
        aeq(n_batch, n_batch_)
        aeq(s_len, s_len_)
        aeq(t_len, t_len_)
        return output, attn
