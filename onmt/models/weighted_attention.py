import math
import copy
import torch
import torch.nn as nn
from onmt.utils.misc import generate_relative_positions_matrix,\
                            relative_matmul

class WeightedAttention(nn.Module):
    """Weighted Attention module from "Weighted Transformer Network for Machine 
    Translation. 

    Args:
        head_count (int): Number of heads
        model_dim (int): Number of dimensions for the embedding. Should be divisible by
            number of heads
        dropout (float): dropout probability
    """
    def __init__(self,head_count,model_dim,dropout=0.1,
                 max_relative_positions=0):
        #TODO: This can be done using inheritance  
        super(WeightedAttention,self).__init__()
        assert model_dim % head_count == 0
        self.model_dim = model_dim
        self.dim_per_head = model_dim // head_count
        self.head_count=head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax=nn.Softmax(dim=1)
        

        self.kappa_w = nn.Parameter(torch.rand(head_count))
        self.kappa_softmax = nn.Softmax(dim=0)        
        # https://arxiv.org/pdf/1803.02155.pdf 
        # Using relative posiitions to get better performance 
        self.max_relative_positions= max_relative_positions
        self.dropout = nn.Dropout(dropout)

        final_linear_per_head = nn.Linear(self.dim_per_head,self.model_dim)
        self.final_linear = _get_clones(final_linear_per_head, head_count)


    def forward(self, key, value, query, mask=None,
                layer_cache=None, attn_type=None):

        # (N, S, E)
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""

            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)
        
        if layer_cache is not None:
            if attn_type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        if self.max_relative_positions > 0 and attn_type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))

        query = shape(query)
        
        # @Reo: Q,K,V is (Batch Size x n_head x seq_len x Dim)
        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        if self.max_relative_positions > 0 and attn_type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        
        # @Reo: Scores (Batch_Size x n_head x seq_len x seq_len)
        # @Reo: Scores = QK^T/d_k
        scores = scores.float()
        
        

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        # self.softmax(dim=1) 
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        # Compute the New values after attention

        context_original = torch.matmul(drop_attn, value)

        #Relative Positional Embeddings
        if self.max_relative_positions > 0 and attn_type == "self":
            context_original += relative_matmul(drop_attn, relations_values,False)
        
        # Context Original - (Batch x N_head x Seq_len x d_h) (10 x 8 x 20 x 64)
        # final_linear is a list of linear layers for each head
        kappa = self.kappa_softmax(self.kappa_w)


        linear_out = torch.cat([self.final_linear[i]\
            (torch.narrow(context_original,1,i,1).contiguous()) \
            for i in range(self.head_count)],1)

        # Concat - (Batch x N_head x Seq_len x d_model) (10 x 8 x 20 x 512)
        out = torch.mul(linear_out.transpose(1,3).contiguous(),kappa)
        out = out.transpose(1,3).contiguous()


        attns = attn \
            .view(batch_size, head_count,
                  query_len, key_len)

        return out, attns
    def update_dropout(self, dropout):
        self.dropout.p = dropout

def _get_clones(module,N):
    # To separate final linear layer into multiple ones per head
    # Number of parameters will still remain the same
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])