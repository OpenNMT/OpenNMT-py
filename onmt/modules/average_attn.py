""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from onmt.utils.misc import aeq

from onmt.utils.transformer_util import PositionwiseFeedForward


class AverageAttention(nn.Module):
    """
    Average Attention module from
    "Accelerating Neural Transformer via an Average Attention Network"
    :cite:`https://arxiv.org/abs/1805.00631`.

    Args:
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, model_dim, dropout=0.1):
        self.model_dim = model_dim

        super(AverageAttention, self).__init__()

        self.average_layer = PositionwiseFeedForward(model_dim, model_dim, dropout)
        self.gating_layer = nn.Linear(model_dim * 2, model_dim * 2)


    def cumulative_average_mask(self, batch_size, inputs_len):
          """
          Builds the mask to compute the cumulative average as described in
          https://arxiv.org/abs/1805.00631 -- Figure 3

          Args:
            inputs_len: length of the inputs.

          Returns:
            A Tensor of shape [batch_size, input_len, input_len]
          """

          triangle = np.tril(np.ones((inputs_len, inputs_len)))
          weights = np.ones((1, inputs_len)) / np.arange(1, inputs_len + 1)
          mask = torch.from_numpy(triangle * weights.T)

          return mask.unsqueeze(0).expand(batch_size, inputs_len, inputs_len)

    def cumulative_average(self, inputs, mask_or_step, layer_cache=None, step=None):
      """
      Computes the cumulative average as described in
      https://arxiv.org/abs/1805.00631 -- Equations (1) (5) (6)

      Args:
        inputs: sequence to average -- Tensor of shape [batch_size, input_len, dimension]
        mask_or_step: if cache is set, this is assumed to be the current step of the
          dynamic decoding. Otherwise, it is the mask matrix used to compute the cumulative average.
        cache: a dictionary containing the cumulative average of the previous step.
      """
      if layer_cache is not None:
        step = mask_or_step
        device = inputs.device
        average_attention = (inputs + step * layer_cache["prev_g"].to(device)) / (step + 1)
        layer_cache["prev_g"] = average_attention
        return average_attention
      else:
        mask = mask_or_step
        return torch.matmul(mask, inputs)

    def forward(self, inputs, mask=None, layer_cache=None, step=None):

        batch_size = inputs.size(0)
        inputs_len = inputs.size(1)

        device = inputs.device
        average_outputs = self.cumulative_average(inputs,
          self.cumulative_average_mask(batch_size, inputs_len).to(device).float() if layer_cache is None else step,
          layer_cache=layer_cache)
        average_outputs = self.average_layer(average_outputs)
        concat = torch.cat((inputs, average_outputs), -1)
        gating_outputs = self.gating_layer(torch.cat((inputs, average_outputs), -1))
        input_gate, forget_gate = torch.split(gating_outputs, int(gating_outputs.size(2)/2), dim=2)
        gating_outputs = torch.sigmoid(input_gate) * inputs + torch.sigmoid(forget_gate) * average_outputs

        return gating_outputs, None