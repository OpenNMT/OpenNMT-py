# -*- coding: utf-8 -*-
"""Average Attention module."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction


def cumulative_average_mask(batch_size: int, inputs_len: int,
                            device: Optional[torch.device] = None) -> Tensor:
    """
    Builds the mask to compute the cumulative average as described in
    :cite:`DBLP:journals/corr/abs-1805-00631` -- Figure 3
    Args:
        batch_size (int): batch size
        inputs_len (int): length of the inputs

    Returns:
        (Tensor):

        * A Tensor of shape ``(batch_size, input_len, input_len)``
    """

    triangle = torch.tril(torch.ones(inputs_len, inputs_len,
                          dtype=torch.float, device=device))
    weights = torch.ones(1, inputs_len, dtype=torch.float, device=device) \
        / torch.arange(1, inputs_len + 1, dtype=torch.float, device=device)
    mask = triangle * weights.transpose(0, 1)

    return mask.unsqueeze(0).expand(batch_size, inputs_len, inputs_len)


def cumulative_average(inputs: Tensor, layer_cache: tuple,
                       mask=None, step=None) -> Tensor:
    """
    Computes the cumulative average as described in
    :cite:`DBLP:journals/corr/abs-1805-00631` -- Equations (1) (5) (6)

    Args:
        inputs (FloatTensor): sequence to average
            ``(batch_size, input_len, dimension)``
        layer_cache: tuple(bool, dict)
        if layer_cahe[0] is True use step otherwise mask
        mask: mask matrix used to compute the cumulative average
        step: current step of the dynamic decoding

    Returns:
        a tensor of the same shape and type as ``inputs``.
    """

    if layer_cache[0]:
        average_attention = (inputs + step *
                             layer_cache[1]['prev_g']) / (step + 1)
        layer_cache[1]['prev_g'] = average_attention
        return average_attention
    else:
        return torch.matmul(mask.to(inputs.dtype), inputs)


class AverageAttention(nn.Module):
    # class AverageAttention(torch.jit.ScriptModule):
    """
    Average Attention module from
    "Accelerating Neural Transformer via an Average Attention Network"
    :cite:`DBLP:journals/corr/abs-1805-00631`.

    Args:
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
       pos_ffn_activation_fn (ActivationFunction):
           activation function choice for PositionwiseFeedForward layer
    """

    def __init__(self, model_dim, dropout=0.1, aan_useffn=False,
                 pos_ffn_activation_fn=ActivationFunction.relu):
        self.model_dim = model_dim
        self.aan_useffn = aan_useffn
        super(AverageAttention, self).__init__()
        if aan_useffn:
            self.average_layer = PositionwiseFeedForward(model_dim, model_dim,
                                                         dropout,
                                                         pos_ffn_activation_fn
                                                         )
        self.gating_layer = nn.Linear(model_dim * 2, model_dim * 2)
        self.layer_cache = False, {'prev_g': torch.tensor([])}

    # @torch.jit.script
    def forward(self, inputs, mask=None, step=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor, FloatTensor):

            * gating_outputs ``(batch_size, input_len, model_dim)``
            * average_outputs average attention
                ``(batch_size, input_len, model_dim)``
        """

        batch_size = inputs.size(0)
        inputs_len = inputs.size(1)
        mask = cumulative_average_mask(batch_size, inputs_len, inputs.device)\
            if not self.layer_cache[0] else None
        average_outputs = cumulative_average(
          inputs, self.layer_cache, mask, step)
        if self.aan_useffn:
            average_outputs = self.average_layer(average_outputs)
        gating_outputs = self.gating_layer(torch.cat((inputs,
                                                      average_outputs), -1))
        input_gate, forget_gate = torch.chunk(gating_outputs, 2, dim=2)
        gating_outputs = torch.sigmoid(input_gate) * inputs + \
            torch.sigmoid(forget_gate) * average_outputs

        return gating_outputs, average_outputs
