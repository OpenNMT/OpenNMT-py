"""
Implementation of double transformer
"""
import torch

from onmt.decoders import TransformerDecoder
from onmt.encoders import TransformerEncoder
from onmt.encoders.encoder import EncoderBase


from torch import Tensor
import torch.nn.functional as f


import torch.nn as nn

from onmt.encoders.encoder import EncoderBase



class DoubleTransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

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
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, opt, embeddings, tg_embeddings=None):
        super(DoubleTransformerEncoder, self).__init__()
        self.first_encoder = TransformerEncoder.from_opt(opt, embeddings)
        self.decoder = TransformerDecoder.from_opt(opt, tg_embeddings)
        self.second_encoder = TransformerEncoder.from_opt(opt, embeddings)
        self.bptt = False

    @classmethod
    def from_opt(cls, opt, embeddings, tg_embeddings=None):
        """Alternate constructor."""
        return cls(opt, embeddings, tg_embeddings)


    def forward(self, src, lengths=None, dec_in=None, bptt=False):
        """See :func:`EncoderBase.forward()`"""
        enc_state, memory_bank, lengths = self.first_encoder(src, lengths)
        if self.bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)

        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=False)
        dec_out = torch.argmax(dec_out, dim=2)
        dec_out.unsqueeze_(-1)
        enc_state2, memory_bank2, lengths2 = self.second_encoder(dec_out, torch.tensor([dec_out.shape[0], dec_out.shape[1]]))
        return enc_state2, memory_bank2, lengths2, dec_out


    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
