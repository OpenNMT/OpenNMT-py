import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from onmt.Utils import aeq
from onmt.modules.UtilClass import BottleLinear, BottleSoftmax


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = BottleLinear(model_dim,
                                        head_count * self.dim_per_head,
                                        bias=False)
        self.linear_values = BottleLinear(model_dim,
                                          head_count * self.dim_per_head,
                                          bias=False)
        self.linear_query = BottleLinear(model_dim,
                                         head_count * self.dim_per_head,
                                         bias=False)
        self.sm = BottleSoftmax()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.res_dropout = nn.Dropout(dropout)

    def forward(self, key, value, query, mask=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        batch, k_len, d = key.size()
        batch_, k_len_, d_ = value.size()
        aeq(batch, batch_)
        aeq(k_len, k_len_)
        aeq(d, d_)
        batch_, q_len, d_ = query.size()
        aeq(batch, batch_)
        aeq(d, d_)
        aeq(self.model_dim % 8, 0)
        if mask is not None:
            batch_, q_len_, k_len_ = mask.size()
            aeq(batch_, batch)
            aeq(k_len_, k_len)
            aeq(q_len_ == q_len)
        # END CHECKS

        def shape_projection(x):
            b, l, d = x.size()
            return x.view(b, l, self.head_count, self.dim_per_head) \
                .transpose(1, 2).contiguous() \
                .view(b * self.head_count, l, self.dim_per_head)

        def unshape_projection(x, q):
            b, l, d = q.size()
            return x.view(b, self.head_count, l, self.dim_per_head) \
                    .transpose(1, 2).contiguous() \
                    .view(b, l, self.head_count * self.dim_per_head)

        residual = query
        key_up = shape_projection(self.linear_keys(key))
        value_up = shape_projection(self.linear_values(value))
        query_up = shape_projection(self.linear_query(query))

        scaled = torch.bmm(query_up, key_up.transpose(1, 2))
        scaled = scaled / math.sqrt(self.dim_per_head)
        bh, l, dim_per_head = scaled.size()
        b = bh // self.head_count
        if mask is not None:

            scaled = scaled.view(b, self.head_count, l, dim_per_head)
            mask = mask.unsqueeze(1).expand_as(scaled)
            scaled = scaled.masked_fill(Variable(mask), -1e18) \
                           .view(bh, l, dim_per_head)
        attn = self.sm(scaled)
        # Return one attn
        top_attn = attn \
            .view(b, self.head_count, l, dim_per_head)[:, 0, :, :] \
            .contiguous()

        drop_attn = self.dropout(self.sm(scaled))

        # values : (batch * 8) x qlen x dim
        out = unshape_projection(torch.bmm(drop_attn, value_up), residual)

        # Residual and layer norm
        ret = self.res_dropout(out)

        # CHECK
        batch_, q_len_, d_ = ret.size()
        aeq(q_len, q_len_)
        aeq(batch, batch_)
        aeq(d, d_)
        # END CHECK
        return ret, top_attn
