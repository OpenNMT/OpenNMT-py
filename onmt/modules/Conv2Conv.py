"""
Implementation of "Convolutional Sequence to Sequence Learning"
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import onmt.modules
from WeightNorm import WN_Conv2d
import torch.nn.functional as F
from torch.autograd import Variable


scale_weight = 0.5 ** 0.5


def shape_transform(x):
    return torch.unsqueeze(torch.transpose(x, 1, 2), 3)


class GatedConv(nn.Module):
    def __init__(self, input_size, width=3, dropout=0.2, nopad=False):
        super(GatedConv, self).__init__()
        self.conv = WN_Conv2d(input_size, 2 * input_size, kernel_size=(width, 1),
                              stride=(1, 1), padding=(width // 2 * (1 - nopad), 0))
        init.xavier_uniform(self.conv.weight, gain=(4 * (1 - dropout))**0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_var, hidden=None):
        x_var = self.dropout(x_var)
        x_var = self.conv(x_var)
        out, gate = x_var.split(x_var.size(1) / 2, 1)
        out = out * F.sigmoid(gate)
        return out


class StackedCNN(nn.Module):
    def __init__(self, n_layers, input_size, width=3, dropout=0.2):
        super(StackedCNN, self).__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(GatedConv(input_size, width, dropout))

    def forward(self, x, hidden=None):
        for conv in self.layers:
            x = x + conv(x)
            x *= scale_weight
        return x


class ConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, layers, dropout, width):
        self.n_layers = layers
        self.hidden_size = hidden_size
        super(ConvEncoder, self).__init__()
        self.linear = nn.Linear(input_size, self.hidden_size)
        self.conv = StackedCNN(self.n_layers, self.hidden_size, width, dropout)

    def forward(self, emb):
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)
        emb_remap = shape_transform(emb_remap)
        outputs = self.conv(emb_remap)

        return outputs.squeeze(3), emb_remap.squeeze(3)


class ConvDecoder(nn.Module):

    def __init__(self, hidden_size, layers, opt):
        self.n_layers = layers
        input_size = opt.tgt_word_vec_size
        self.width = opt.width
        self.hidden_size = hidden_size
        super(ConvDecoder, self).__init__()
        self.linear = nn.Linear(input_size, self.hidden_size)
        self.conv_layers = nn.ModuleList()
        if hasattr(opt, 'gpuid'):
            self.cuda = len(opt.gpuid) >= 1
        elif hasattr(opt, 'gpu'):
            self.cuda = opt.gpu > -1
        else:
            self.cuda = False
        for i in range(self.n_layers):
            self.conv_layers.append(
                GatedConv(self.hidden_size, opt.width, opt.dropout, True))

        self.attn_layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                onmt.modules.ConvAttention(self.hidden_size))

    def forward(self, target_emb, encoder_out_t, encoder_out_c):
        emb_reshape = target_emb.contiguous().view(
            target_emb.size(0) * target_emb.size(1), -1)
        linear_out = self.linear(emb_reshape)
        x = linear_out.view(target_emb.size(0), target_emb.size(1), -1)
        x = shape_transform(x)

        pad = Variable(torch.zeros(x.size(0), x.size(1), self.width - 1, 1))
        if self.cuda:
            pad = pad.cuda()
        base_target_emb = x

        for conv, attention in zip(self.conv_layers, self.attn_layers):
            new_target_input = torch.cat([pad, x], 2)
            out = conv(new_target_input)
            c, attn = attention(base_target_emb, out,
                                encoder_out_t, encoder_out_c)
            x = (x + (c + out) * scale_weight) * scale_weight
        return x.squeeze(3).transpose(1, 2), attn
