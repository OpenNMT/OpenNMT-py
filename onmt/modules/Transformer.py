"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np
import onmt.modules
from onmt.modules import aeq


MAX_SIZE = 5000


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
    def __init__(self, size, dropout,
                 head_count=8, hidden_size=2048):
        """
        Args:
            size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
            droput(float): dropout probability(0-1.0).
            head_count(int): the number of head for MultiHeadedAttention.
            hidden_size(int): the second-layer of the PositionwiseFeedForward.
        """
        super(TransformerEncoder, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            head_count, size, p=dropout)
        self.feed_forward = PositionwiseFeedForward(size,
                                                    hidden_size,
                                                    dropout)

    def forward(self, input, mask):
        mid, _ = self.self_attn(input, input, input, mask=mask)
        out = self.feed_forward(mid)
        return out


class TransformerDecoder(nn.Module):
    """
    The Transformer Decoder from paper "Attetion is all you need".
    """
    def __init__(self, size, dropout,
                 head_count=8, hidden_size=2048):
        """
        Args:
            size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
            droput(float): dropout probability(0-1.0).
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
        self.mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoder, so
        # it gets TransformerDecoder's cuda behavior automatically.
        self.register_buffer('mask', self.mask)

    def forward(self, input, context, src_pad_mask, tgt_pad_mask):
        # Args Checks
        input_batch, input_len, _ = input.size()
        contxt_batch, contxt_len, _ = context.size()
        aeq(input_batch, contxt_batch)

        src_batch, t_len, s_len = src_pad_mask.size()
        tgt_batch, t_len_, t_len__ = tgt_pad_mask.size()
        aeq(input_batch, contxt_batch, src_batch, tgt_batch)
        aeq(t_len, t_len_, t_len__, input_len)
        aeq(s_len, contxt_len)
        # END Args Checks

        dec_mask = torch.gt(tgt_pad_mask + self.mask[:, :tgt_pad_mask.size(1),
                            :tgt_pad_mask.size(1)]
                            .expand_as(tgt_pad_mask), 0)
        query, attn = self.self_attn(input, input, input, mask=dec_mask)
        mid, attn = self.context_attn(context, context, query,
                                      mask=src_pad_mask)
        output = self.feed_forward(mid)

        # CHECKS
        output_batch, output_len, _ = output.size()
        aeq(input_len, output_len)
        aeq(contxt_batch, output_batch)

        n_batch_, t_len_, s_len_ = attn.size()
        aeq(input_batch, n_batch_)
        aeq(contxt_len, s_len_)
        aeq(input_len, t_len_)
        # END CHECKS

        return output, attn

    def _get_attn_subsequent_mask(self, size):
        ''' Get an attention mask to avoid using the subsequent info.'''
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask
