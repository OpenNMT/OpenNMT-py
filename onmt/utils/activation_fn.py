import torch
import torch.nn as nn
import math


def get_activation_fn(activation):
    """Return an activation function Module according to its name."""
    if activation is 'gelu':
        fn = GELU()
    elif activation is 'relu':
        fn = nn.ReLU()
    elif activation is 'tanh':
        fn = nn.Tanh()
    else:
        raise ValueError("Please pass a valid \
                          activation function")
    return fn


"""
Adapted from huggingface implementation to reproduce the result
https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/modeling_bert.py
"""
class GELU(nn.Module):
    """ Implementation of the gelu activation function
        :cite:`DBLP:journals/corr/HendrycksG16`

        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi)
                    * (x + 0.044715 * torch.pow(x, 3))))

        Examples::
        >>> m = GELU()
        >>> inputs = torch.randn(2)
        >>> outputs = m(inputs)
    """
    def forward(self, x):
        gelu = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return gelu
