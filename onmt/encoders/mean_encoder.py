"""Define a minimal encoder."""
from onmt.encoders.encoder import EncoderBase
from onmt.utils.misc import sequence_mask
import torch


class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(opt.enc_layers, embeddings)

    def forward(self, src, src_len=None):
        """See :func:`EncoderBase.forward()`"""

        emb = self.embeddings(src)
        batch, _, emb_dim = emb.size()

        if src_len is not None:
            # we avoid padding while mean pooling
            mask = (~sequence_mask(src_len)).float()
            mask = mask / src_len.unsqueeze(1).float()
            mean = torch.bmm(mask.unsqueeze(1), emb).squeeze(1)
        else:
            mean = emb.mean(1)

        mean = mean.expand(self.num_layers, batch, emb_dim)
        enc_out = emb
        enc_final_hs = (mean, mean)
        return enc_out, enc_final_hs, src_len
