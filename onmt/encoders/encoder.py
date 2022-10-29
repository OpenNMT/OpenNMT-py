"""Base class for encoders and generic multi encoders."""

import torch.nn as nn


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :class:`onmt.Models.NMTModel`.

    """

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        raise NotImplementedError

    def forward(self, src, src_len=None):
        """
        Args:
            src (LongTensor):
               padded sequences of sparse indices ``(batch, src_len, nfeat)``
            src_len (LongTensor): length of each sequence ``(batch,)``

        Returns:
            (FloatTensor, FloatTensor, FloatTensor):

            * enc_out (encoder output used for attention),
              ``(batch, src_len, hidden_size)``
              for bidirectional rnn last dimension is 2x hidden_size
            * enc_final_hs: encoder final hidden state
              ``(num_layersxdir, batch, hidden_size)``
              In the case of LSTM this is a tuple.
            * src_len (batch)
        """

        raise NotImplementedError
