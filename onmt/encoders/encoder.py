"""Base class for encoders and generic multi encoders."""

import torch.nn as nn


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :class:`onmt.Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Memory_Bank]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        raise NotImplementedError

    def forward(self, src, lengths=None):
        """
        Args:
            src (LongTensor):
               padded sequences of sparse indices ``(src_len, batch, nfeat)``
            lengths (LongTensor): length of each sequence ``(batch,)``

        Returns:
            (FloatTensor, FloatTensor, FloatTensor):

            * final encoder state, used to initialize decoder
            * memory bank for attention, ``(src_len, batch, hidden)``
            * lengths
        """

        raise NotImplementedError
