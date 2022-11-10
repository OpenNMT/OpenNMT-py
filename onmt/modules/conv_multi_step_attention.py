""" Multi Step Attention for CNN """
import torch
import torch.nn as nn
import torch.nn.functional as F


SCALE_WEIGHT = 0.5 ** 0.5


def seq_linear(linear, x):
    """ linear transform for 3-d tensor """
    batch, hidden_size, length, _ = x.size()
    h = linear(torch.transpose(x, 1, 2).contiguous().view(
        batch * length, hidden_size))
    return torch.transpose(h.view(batch, length, hidden_size, 1), 1, 2)


class ConvMultiStepAttention(nn.Module):
    """
    Conv attention takes a key matrix, a value matrix and a query vector.
    Attention weight is calculated by key matrix with the query vector
    and sum on the value matrix. And the same operation is applied
    in each decode conv layer.
    """

    def __init__(self, input_size):
        super(ConvMultiStepAttention, self).__init__()
        self.linear_in = nn.Linear(input_size, input_size)
        self.mask = None

    def apply_mask(self, mask):
        """ Apply mask """
        self.mask = mask

    def forward(self, base_target_emb, input_from_dec, encoder_out_top,
                encoder_out_combine):
        """
        Args:
            base_target_emb: target emb tensor
                ``(batch, channel, height, width)``
            input_from_dec: output of dec conv
                ``(batch, channel, height, width)``
            encoder_out_top: the key matrix for calc of attention weight,
                which is the top output of encode conv
            encoder_out_combine:
                the value matrix for the attention-weighted sum,
                which is the combination of base emb and top output of encode
        """

        preatt = seq_linear(self.linear_in, input_from_dec)
        target = (base_target_emb + preatt) * SCALE_WEIGHT
        target = torch.squeeze(target, 3)
        target = torch.transpose(target, 1, 2)

        pre_attn = torch.bmm(target, encoder_out_top)

        if self.mask is not None:
            pre_attn.data.masked_fill_(self.mask, -float('inf'))

        attn = F.softmax(pre_attn, dim=2)

        context_output = torch.bmm(
            attn, torch.transpose(encoder_out_combine, 1, 2))
        context_output = torch.transpose(
            torch.unsqueeze(context_output, 3), 1, 2)
        return context_output, attn
