"""
Conv attention takes a matrix, a query vector and a value vector.
calculate attention weight by query vector and sum on value vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

scale_weight = 0.5 ** 0.5


def seq_linear(linear, x):
    batch, hidden_size, length, _ = x.size()
    h = linear(torch.transpose(x, 1, 2).contiguous().view(
        batch * length, hidden_size))
    return torch.transpose(h.view(batch, length, hidden_size, 1), 1, 2)


class ConvAttention(nn.Module):
    def __init__(self, input_size):
        super(ConvAttention, self).__init__()
        self.linear_in = nn.Linear(input_size, input_size)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, base_target_emb, input, encoder_out_t, encoder_out_c):
        preatt = seq_linear(self.linear_in, input)
        target = (base_target_emb + preatt) * scale_weight
        target = torch.squeeze(target, 3)
        target = torch.transpose(target, 1, 2)
        pre_a = torch.bmm(target, encoder_out_t)

        if self.mask is not None:
            pre_a.data.masked_fill_(self.mask, -float('inf'))

        attn = F.softmax(pre_a)
        contextOutput = torch.bmm(attn, torch.transpose(encoder_out_c, 1, 2))
        contextOutput = torch.transpose(
            torch.unsqueeze(contextOutput, 3), 1, 2)
        return contextOutput, attn
