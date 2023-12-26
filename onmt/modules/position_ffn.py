"""Position feed-forward network from "Attention is All You Need"."""

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from onmt.modules.rmsnorm import RMSNorm
from torch.nn.utils import skip_init
from torch.distributed import all_reduce


class ActivationFunction(object):
    relu = "relu"
    gelu = "gelu"
    silu = "silu"
    gated_gelu = "gated-gelu"


# for silu, see: https://arxiv.org/pdf/2002.05202.pdf
ACTIVATION_FUNCTIONS = {
    ActivationFunction.relu: F.relu,
    ActivationFunction.gelu: F.gelu,
    ActivationFunction.silu: F.silu,
    ActivationFunction.gated_gelu: F.gelu,
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
        norm_eps=1e-6,
        use_ckpting=[],
        parallel_gpu=1,
    ):
        super(PositionwiseFeedForward, self).__init__()
        assert (
            d_ff % parallel_gpu == 0
        ), "Model intermediate ffn size must be divisible by the number of partitions"
        self.w_1 = skip_init(
            nn.Linear,
            in_features=d_model,
            out_features=d_ff // parallel_gpu,
            bias=add_ffnbias,
        )
        self.w_2 = skip_init(
            nn.Linear,
            in_features=d_ff // parallel_gpu,
            out_features=d_model,
            bias=add_ffnbias,
        )
        if layer_norm == "standard" and not parallel_residual:
            self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)
        elif layer_norm == "rms" and not parallel_residual:
            self.layer_norm = RMSNorm(d_model, eps=norm_eps)
        elif not parallel_residual:
            raise ValueError(f"{layer_norm} layer norm type is not supported")
        self.parallel_residual = parallel_residual
        self.dropout_p = dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.activation = ACTIVATION_FUNCTIONS[activation_fn]
        if activation_fn == "silu" or activation_fn == "gated-gelu":
            self.w_3 = skip_init(
                nn.Linear,
                in_features=d_model,
                out_features=d_ff // parallel_gpu,
                bias=add_ffnbias,
            )
        else:
            self.w_3 = None
        self.maybe_ckpt = checkpoint if "ffn" in use_ckpting else lambda f, x: f(x)
        self.parallel_gpu = parallel_gpu

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
            norm_x = x.clone()
        inter = self.maybe_ckpt(self.w_1, norm_x)
        inter = self.activation(inter)
        if self.w_3 is not None:
            inter.mul_(self.maybe_ckpt(self.w_3, norm_x))
        if self.dropout_p > 0:
            inter = self.dropout_1(inter)
        inter = self.maybe_ckpt(self.w_2, inter)
        if self.dropout_p > 0:
            inter = self.dropout_2(inter)

        if self.parallel_gpu > 1:
            all_reduce(inter)

        return inter + x

    def update_dropout(self, dropout):
        self.dropout_1.p = dropout
        self.dropout_2.p = dropout
