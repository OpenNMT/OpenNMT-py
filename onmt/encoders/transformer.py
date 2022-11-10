"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction
from onmt.utils.misc import sequence_mask


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0,
                 pos_ffn_activation_fn=ActivationFunction.relu,
                 add_qkvbias=False):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions,
            attn_type="self", add_qkvbias=add_qkvbias)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout,
                                                    pos_ffn_activation_fn)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, layer_in, mask):
        """
        Args:
            layer_in (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):
            * layer_out ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(layer_in)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask)
        layer_out = self.dropout(context) + layer_in
        layer_out = self.feed_forward(layer_out)
        return layer_out

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * enc_out ``(batch_size, src_len, model_dim)``
        * encoder final state: None in the case of Transformer
        * src_len ``(batch_size)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions,
                 pos_ffn_activation_fn=ActivationFunction.relu,
                 add_qkvbias=False):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions,
                pos_ffn_activation_fn=pos_ffn_activation_fn,
                add_qkvbias=add_qkvbias)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_hid_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
            add_qkvbias=opt.add_qkvbias
        )

    def forward(self, src, src_len=None):
        """See :func:`EncoderBase.forward()`"""
        enc_out = self.embeddings(src)
        mask = ~sequence_mask(src_len).unsqueeze(1)
        mask = mask.unsqueeze(1)
        mask = mask.expand(-1, -1, mask.size(3), -1)
        # mask is now (batch x 1 x slen x slen)
        # 1 to be expanded to number of heads in MHA
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            enc_out = layer(enc_out, mask)
        enc_out = self.layer_norm(enc_out)

        return enc_out, None, src_len

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
