"""Position feed-forward network from "Attention is All You Need"."""

import torch.nn as nn

from onmt.utils import get_activation_fn


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
        activation (str): activation function to use. ['relu', 'gelu']
        is_bert (bool): default False. When set True,
                        layer_norm will be performed on the
                        direct connection of residual block.
    """

    def __init__(self, d_model, d_ff, dropout=0.1,
                 activation='relu', is_bert=False):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(
            d_model, eps=1e-12 if is_bert else 1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)
        self.dropout_2 = nn.Dropout(dropout)
        self.is_bert = is_bert

    def residual(self, output, x):
        """A Residual connection.

        Official BERT perform residual connection on layer normed input.
        BERT's layer_norm is done before pass into next block while onmt's
        layer_norm is performed at the begining.
        """

        maybe_norm = self.layer_norm(x) if self.is_bert else x
        return output + maybe_norm

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        inter = self.dropout_1(self.activation(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return self.residual(output, x)

    def update_dropout(self, dropout):
        self.dropout_1.p = dropout
        self.dropout_2.p = dropout
