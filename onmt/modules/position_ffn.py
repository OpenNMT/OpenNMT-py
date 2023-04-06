"""Position feed-forward network from "Attention is All You Need"."""


import torch.nn as nn
import torch.nn.functional as F
from onmt.modules.rmsnorm import RMSNorm


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
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
        activation_fn (ActivationFunction): activation function used.
        layer_norm (string): 'standard' or 'rms'
    """

    def __init__(self, d_model, d_ff, dropout=0.1,
                 activation_fn=ActivationFunction.relu,
                 layer_norm='standard'):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        if layer_norm == 'standard':
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        elif layer_norm == 'rms':
            self.layer_norm = RMSNorm(d_model, eps=1e-6)
        else:
            raise ValueError(f'{layer_norm} layer norm type is not supported')
        self.dropout_1 = nn.Dropout(dropout)
        self.activation = ACTIVATION_FUNCTIONS[activation_fn]
        self.dropout_2 = nn.Dropout(dropout)
        if activation_fn == 'silu':
            self.w_3 = nn.Linear(d_model, d_ff)
        else:
            self.w_3 = None

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """
        if self.w_3 is None:
            inter = self.dropout_1(self.activation(
                self.w_1(self.layer_norm(x))))
        else:
            inter = self.dropout_1(
                self.activation(self.w_1(self.layer_norm(x)))
                * self.w_3(self.layer_norm(x)))
        output = self.dropout_2(self.w_2(inter))
        return output + x

    def update_dropout(self, dropout):
        self.dropout_1.p = dropout
        self.dropout_2.p = dropout
