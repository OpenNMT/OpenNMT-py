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

    def forward(self, src, lengths=None):
        """
        Args:
            src (LongTensor):
               padded sequences of sparse indices ``(batch, src_len, nfeat)``
            lengths (LongTensor): length of each sequence ``(batch,)``

        Returns:
            (FloatTensor, FloatTensor, FloatTensor):

            * output (for attention), ``(batch, src_len, hidden_size)``
              for bidirectional rnn last dimension is 2x hidden_size
            * final_hidden_state ``(num_layersxdir, batch, hidden_size)``
              (used to initialize decoder in RNNs)
              In the case of LSTM this is a tuple.
            * lengths (batch)
        """

        raise NotImplementedError
