"""Position feed-forward network from "Attention is All You Need"."""


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from onmt.modules.rmsnorm import RMSNorm
from torch.nn.utils import skip_init


class ActivationFunction(object):
    relu = "relu"
    gelu = "gelu"
    silu = "silu"


# for silu, see: https://arxiv.org/pdf/2002.05202.pdf
ACTIVATION_FUNCTIONS = {
    ActivationFunction.relu: F.relu,
    ActivationFunction.gelu: F.gelu,
    ActivationFunction.silu: F.silu,
}


class PositionwiseFeedForward(nn.Module):
    """A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
        activation_fn (ActivationFunction): activation function used.
        layer_norm (string): 'standard' or 'rms'
    """

    def __init__(
        self,
        d_model,
        d_ff,
        dropout=0.1,
        activation_fn=ActivationFunction.relu,
        add_ffnbias=True,
        parallel_residual=False,
        layer_norm="standard",
        use_ckpting=[],
    ):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = skip_init(
            nn.Linear, in_features=d_model, out_features=d_ff, bias=add_ffnbias
        )
        self.w_2 = skip_init(
            nn.Linear, in_features=d_ff, out_features=d_model, bias=add_ffnbias
        )
        if layer_norm == "standard" and not parallel_residual:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        elif layer_norm == "rms" and not parallel_residual:
            self.layer_norm = RMSNorm(d_model, eps=1e-6)
        elif not parallel_residual:
            raise ValueError(f"{layer_norm} layer norm type is not supported")
        self.parallel_residual = parallel_residual
        self.dropout_1 = nn.Dropout(dropout)
        self.activation = ACTIVATION_FUNCTIONS[activation_fn]
        self.dropout_2 = nn.Dropout(dropout)
        if activation_fn == "silu":
            self.w_3 = skip_init(
                nn.Linear, in_features=d_model, out_features=d_ff, bias=add_ffnbias
            )
        else:
            self.w_3 = None
        self.maybe_ckpt = checkpoint if "ffn" in use_ckpting else lambda f, x: f(x)

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """
        if not self.parallel_residual:
            norm_x = self.layer_norm(x)
        else:
            norm_x = x
        inter = self.maybe_ckpt(self.w_1, norm_x)
        inter = self.activation(inter)
        if self.w_3 is not None:
            inter.mul_(self.maybe_ckpt(self.w_3, norm_x))
        inter = self.dropout_1(inter)
        inter = self.maybe_ckpt(self.w_2, inter)
        inter = self.dropout_2(inter)
        return inter + x

    def update_dropout(self, dropout):
        self.dropout_1.p = dropout
        self.dropout_2.p = dropout
