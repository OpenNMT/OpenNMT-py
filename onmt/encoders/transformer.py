"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

import onmt
from onmt.encoders.encoder import EncoderBase
# from onmt.utils.misc import aeq
from onmt.modules.position_ffn import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        size(int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        droput(float): dropout probability(0-1.0).
        head_count(int): the number of head for MultiHeadedAttention.
        hidden_size(int): the second-layer of the PositionwiseFeedForward.
    """

    def __init__(self, size, dropout,
                 head_count=8, hidden_size=2048):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            head_count, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size,
                                                    hidden_size,
                                                    dropout)
        self.layer_norm = onmt.modules.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs: [ batch_size, src_len, model_dim ]
            mask: [ batch_size, src_len, src_len ]

        Returns:
            outputs: [ batch_size, src_len, model_dim ]
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        hidden_size (int): number of hidden units
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    Returns:
        embeddings: [ src_len, batch_size, model_dim ]
        memory_bank: [ src_len, batch_size, model_dim ]
    """

    def __init__(self, num_layers, hidden_size,
                 dropout, embeddings):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(hidden_size, dropout)
             for _ in range(num_layers)])
        self.layer_norm = onmt.modules.LayerNorm(hidden_size)

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        # CHECKS - commented for performance issue
        # out_batch, out_len, _ = out.size()
        # w_batch, w_len = words.size()
        # aeq(out_batch, w_batch)
        # aeq(out_len, w_len)
        # END CHECKS

        # Make mask.
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous()
