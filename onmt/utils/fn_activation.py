import torch
import torch.nn as nn
import math


class GELU(nn.Module):
    """ Implementation of the gelu activation function

        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi)
                    * (x + 0.044715 * torch.pow(x, 3))))
        see https://arxiv.org/abs/1606.08415

        Examples::
        >>> m = GELU()
        >>> inputs = torch.randn(2)
        >>> outputs = m(inputs)
    """
    def forward(self, x):
        gelu = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return gelu
