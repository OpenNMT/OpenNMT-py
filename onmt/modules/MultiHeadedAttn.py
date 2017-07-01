import torch
import torch.nn as nn
from torch.autograd import Variable
from onmt.modules import MatrixTree
from onmt.modules.Util import Bottle2, BottleLinear, BottleLayerNorm, \
    BottleSoftmax
import math


class BottleStruct(Bottle2, MatrixTree):
    pass


class MultiHeadedAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, p=0.1, use_struct=False):
        self.d_k = d_model // n_head

        super(MultiHeadedAttention, self).__init__()
        heads = self.heads = n_head
        self.linear_keys = BottleLinear(d_model, heads * self.d_k, bias=False)
        self.linear_values = BottleLinear(d_model, heads * self.d_k,
                                          bias=False)
        self.linear_query = BottleLinear(d_model, heads * self.d_k, bias=False)

        if use_struct:
            self.sm = BottleStruct()
        else:
            self.sm = BottleSoftmax()

        self.activation = nn.ReLU()
        self.layer_norm = BottleLayerNorm(d_model)
        self.dropout = nn.Dropout(p)
        self.res_dropout = nn.Dropout(p)

    def forward(self, key, value, query, mask=None):
        residual = query

        def shape(x):
            return x.view(x.size(0), x.size(1), self.heads, self.d_k) \
                    .transpose(1, 2) \
                    .contiguous() \
                    .view(x.size(0) * self.heads, x.size(1), self.d_k)

        def unshape(x):
            return x.view(x.size(0) // self.heads, self.heads,
                          x.size(1), x.size(2))

        def smash(x):
            return x.view(x.size(0) * self.heads, x.size(2), x.size(3))

        key_up = shape(self.linear_keys(key))
        value_up = shape(self.linear_values(value))
        query_up = shape(self.linear_query(query))

        scaled = torch.bmm(query_up, key_up.transpose(1, 2))
        scaled = scaled / math.sqrt(self.d_k)

        if mask is not None:
            scaled = unshape(scaled)
            mask = mask.unsqueeze(1).expand_as(scaled)
            scaled = scaled.masked_fill(Variable(mask), -float('inf'))
            scaled = smash(scaled)
        attn = self.dropout(self.sm(scaled))

        # values : (batch * 8) x qlen x dim
        out = torch.bmm(attn, value_up)

        out = out.view(query.size(0), self.heads, query.size(1), self.d_k) \
                 .transpose(1, 2).contiguous() \
                 .view(query.size(0), query.size(1), self.heads * self.d_k)

        # Residual and layer norm
        res = self.res_dropout(out) + residual
        return self.layer_norm(res), attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = BottleLinear(d_hid, d_inner_hid)
        self.w_2 = BottleLinear(d_inner_hid, d_hid)
        self.layer_norm = BottleLayerNorm(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x))
        output = self.w_2(output)
        output = self.dropout(output)
        return self.layer_norm(output + residual)
