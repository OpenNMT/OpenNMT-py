import torch
import torch.nn as nn
from onmt.encoders.transformer import TransformerEncoderLayer


class BertEncoder(nn.Module):
    """BERT Encoder: A Transformer Encoder with BertLayerNorm and BertPooler.
    :cite:`DBLP:journals/corr/abs-1810-04805`

    Args:
       embeddings (onmt.modules.BertEmbeddings): embeddings to use
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
    """

    def __init__(self, embeddings, num_layers=12, d_model=768, heads=12,
                 d_ff=3072, dropout=0.1, attention_dropout=0.1,
                 max_relative_positions=0):
        super(BertEncoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.heads = heads
        self.dropout = dropout
        # Feed-Forward size should be 4*d_model as in paper
        self.d_ff = d_ff

        self.embeddings = embeddings
        # Transformer Encoder Block
        self.encoder = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff,
             dropout, attention_dropout,
             max_relative_positions=max_relative_positions,
             activation='gelu', is_bert=True) for _ in range(num_layers)])

        self.layer_norm = BertLayerNorm(d_model, eps=1e-12)
        self.pooler = BertPooler(d_model)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            embeddings,
            opt.layers,
            opt.word_vec_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            opt.max_relative_positions)

    def forward(self, input_ids, token_type_ids=None, input_mask=None,
                output_all_encoded_layers=False):
        """
        Args:
            input_ids (Tensor): ``(B, S)``, padding ids=0
            token_type_ids (Tensor): ``(B, S)``, A(0), B(1), pad(0)
            input_mask (Tensor): ``(B, S)``, 1 for masked (padding)
            output_all_encoded_layers (bool): if out contain all hidden layer
        Returns:
            all_encoder_layers (list of Tensor): ``(B, S, H)``, token level
            pooled_output (Tensor): ``(B, H)``, sequence level
        """

        # OpenNMT waiting for mask of size [B, 1, T],
        # while in MultiHeadAttention part2 -> [B, 1, 1, T]
        if input_mask is None:
            # shape: 2D tensor [batch, seq]
            padding_idx = self.embeddings.word_padding_idx
            # shape: 2D tensor [batch, seq]: 1 for tokens, 0 for paddings
            input_mask = input_ids.data.eq(padding_idx)
        # [batch, seq] -> [batch, 1, seq]
        attention_mask = input_mask.unsqueeze(1)

        # embedding vectors: [batch, seq, hidden_size]
        out = self.embeddings(input_ids, token_type_ids)
        all_encoder_layers = []
        for layer in self.encoder:
            out = layer(out, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(self.layer_norm(out))
        out = self.layer_norm(out)
        if not output_all_encoded_layers:
            all_encoder_layers.append(out)
        pooled_out = self.pooler(out)
        return all_encoder_layers, pooled_out

    def update_dropout(self, dropout):
        self.dropout = dropout
        self.embeddings.update_dropout(dropout)
        for layer in self.encoder:
            layer.update_dropout(dropout)


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        """A pooling block (Linear layer followed by Tanh activation).

        Args:
            hidden_size (int): size of hidden layer.
        """

        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation_fn = nn.Tanh()

    def forward(self, hidden_states):
        """hidden_states[:, 0, :] --> {Linear, Tanh} --> Returns.

        Args:
            hidden_states (Tensor): last layer's hidden_states, ``(B, S, H)``
        Returns:
            pooled_output (Tensor): transformed output of last layer's hidden
        """

        first_token_tensor = hidden_states[:, 0, :]  # [batch, d_model]
        pooled_output = self.activation_fn(self.dense(first_token_tensor))
        return pooled_output


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Layernorm module in the TF style(epsilon inside the square root).
        https://www.tensorflow.org/api_docs/python/tf/contrib/layers/layer_norm.

        Args:
            hidden_size (int): size of hidden layer.
        """

        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        """layer normalization is perform on input x."""

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
