import torch
import torch.nn as nn
import torch.nn.init as init
import onmt.modules
import numpy as np
from torch.autograd import Variable
from onmt.modules import LayerNorm, DependencyTree
from onmt.modules.Util import LayerNorm, Bottle, Bottle2, BottleLinear, BottleLayerNorm, BottleSoftmax
import math

class BottleStruct(Bottle2, DependencyTree):
    pass

class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''
    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax()

    def forward(self, q, k, v, attn_mask=None):
        
        _, len_q, _ = q.size()
        _, len_k, _ = k.size()

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn.view(-1, len_k)).view(-1, len_q, len_k)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class MultiHeadedAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, p=0.1, use_struct=False):

        dropout = p
        dim_size = d_model
        self.d_k = d_k = d_model // n_head
        d_v = d_model // n_head
        super(MultiHeadedAttention, self).__init__()
        # self.w_qs = nn.ModuleList([
        #     Linear(d_model, d_k, bias=False) for _ in range(n_head)])
        # self.w_ks = nn.ModuleList([
        #     Linear(d_model, d_k, bias=False) for _ in range(n_head)])
        # self.w_vs = nn.ModuleList([
        #     Linear(d_model, d_v, bias=False) for _ in range(n_head)])

        # self.attention = ScaledDotProductAttention(d_model)
        # # self.layer_norm = LayerNorm(d_model)
        # self.proj = Linear(n_head*d_v, d_model)
        # # self.dropout = nn.Dropout(dropout)

        # stuff
        heads = self.heads = n_head
        self.linear_keys = BottleLinear(dim_size, heads * self.d_k, bias=False)
        self.linear_values = BottleLinear(dim_size, heads * self.d_k, bias=False)
        self.linear_query = BottleLinear(dim_size, heads * self.d_k, bias=False)
        if use_struct:
            self.sm = BottleStruct()
        else:
            self.sm = BottleSoftmax()
        self.activation = nn.ReLU()
        self.layer_norm = BottleLayerNorm(dim_size)
        self.dropout = nn.Dropout(p)
        self.res_dropout = nn.Dropout(p)

    def forward(self, key, value, query, mask=None):
        residual = query
        if False:
            k, v, q = key, value, query
            outputs, attns = [], []
            mb_size, len_q, d_model = q.size()
            mb_size, len_k, d_model = k.size()
            mb_size, len_v, d_model = v.size()

            # Old
            for w_qi, w_ki, w_vi in zip(self.w_qs, self.w_ks, self.w_vs):
                q_i = w_qi(q.view(-1, d_model)).view((mb_size, len_q, -1))
                k_i = w_ki(k.view(-1, d_model)).view((mb_size, len_k, -1))
                v_i = w_vi(v.view(-1, d_model)).view((mb_size, len_v, -1))
                output, attn = self.attention(q_i, k_i, v_i, attn_mask=mask)
                outputs += [output]
                attns += [attn]

            outputs = torch.cat(outputs, 2)
            outputs = self.proj(outputs.view(-1, outputs.size(2))).view_as(residual)
            outputs = self.dropout(outputs)

            return self.layer_norm(outputs + residual), attns


        # Newi
        if True:
            # print(key.size())
            # print(query.size())
            def shape(x):
                return x.view(x.size(0), x.size(1), self.heads, self.d_k).transpose(1, 2) \
                        .contiguous().view(x.size(0) * self.heads, x.size(1), self.d_k)

            def unshape(x):
                return x.view(x.size(0)//self.heads, self.heads, x.size(1), x.size(2))
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
                # scaled.data.masked_fill_(mask, -float('inf'))
                # print(scaled.size())
                scaled = smash(scaled)
            attn = self.dropout(self.sm(scaled))
            # attn = self.sm(scaled)
            # values : (batch * 8) x qlen x dim
            out = torch.bmm(attn, value_up)

            out = out.view(query.size(0), self.heads, query.size(1), self.d_k) \
                     .transpose(1, 2).contiguous() \
                     .view(query.size(0), query.size(1), self.heads * self.d_k)
            # Residual and layer norm
            res = self.res_dropout(out) + query
            return self.layer_norm(res), attn

        # def shape(x):
        #     return x.view(x.size(0), x.size(1), self.heads, self.d_k).transpose(1, 2) \
        #             .contiguous().view(x.size(0) * self.heads, x.size(1), self.d_k)
        
        # key_up = shape(self.linear_keys(key))
        # value_up = shape(self.linear_values(value))
        # query_up = shape(self.linear_query(query))

        # scaled = torch.bmm(query_up, key_up.transpose(1, 2)) 
        # scaled = scaled / math.sqrt(self.d_k)

        # if mask:
        #     scaled = scaled + self.mask[:, :query.size(1), :key.size(1)].expand_as(scaled)


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = BottleLinear(d_hid, d_inner_hid)
        self.w_2 = BottleLinear(d_inner_hid, d_hid) 
        # self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1) # position-wise
        # self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1) # position-wise
        self.layer_norm = BottleLayerNorm(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x))
        output = self.w_2(output)
        output = self.dropout(output)

        return self.layer_norm(output + residual)


# class MultiHeadedAttention(nn.Module):
#     def __init__(self, heads, dim_size, p=0.1):
#         super(MultiHeadedAttention, self).__init__()
#         self.heads = heads
#         self.d_k = dim_size // heads
#         self.linear_keys = BottleLinear(dim_size, heads * self.d_k, bias=False)
#         self.linear_values = BottleLinear(dim_size, heads * self.d_k, bias=False)
#         self.linear_query = BottleLinear(dim_size, heads * self.d_k, bias=False) 
#         self.sm = BottleSoftmax()
#         self.activation = nn.ReLU()
#         self.layer_norm = LayerNorm(dim_size)
#         self.dropout = p
#         self.mask = torch.FloatTensor(1, 1000, 1000).fill_(0)
#         for i in range(1000):
#             for j in range(1000):
#                 if j > i:
#                     self.mask[:, i, j] = -1e20
#         self.mask = Variable(self.mask.cuda())
                
#     def forward(self, key, value, query, mask=False):
#         """
#         Args:
#             key (FloatTensor):   batch x len x dim
#             value (FloatTensor): batch x len x dim
#             query (FloatTensor): batch x qlen x dim

#         Return: 
#             Res (FloatTensor): batch x qlen x dim
#         """
#         values = []
#         def shape(x):
#             return x.view(x.size(0), x.size(1), self.heads, self.d_k).transpose(1, 2) \
#                     .contiguous().view(x.size(0) * self.heads, x.size(1), self.d_k)
        
#         key_up = shape(self.linear_keys(key))
#         value_up = shape(self.linear_values(value))
#         query_up = shape(self.linear_query(query))

#         scaled = torch.bmm(query_up, key_up.transpose(1, 2)) 
#         scaled = scaled / math.sqrt(self.d_k)

#         if mask:
#             scaled = scaled + self.mask[:, :query.size(1), :key.size(1)].expand_as(scaled)
#         attn = F.dropout(self.sm(scaled), p=self.dropout)
                
#         # values : (batch * 8) x qlen x dimn
#         out = torch.bmm(attn, value_up)
#         out = out.view(query.size(0), self.heads, query.size(1), self.d_k) \
#                  .transpose(1, 2).contiguous() \
#                  .view(query.size(0), query.size(1), self.heads * self.d_k)
        
#         # Residual and layer norm
#         res = out + query
#         return self.layer_norm(res.contiguous().view(-1, res.size(2))).contiguous().view_as(res).contiguous(), attn
        
