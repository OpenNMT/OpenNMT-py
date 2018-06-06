import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.modules.UtilClass import LayerNorm
from onmt.Models import DecoderState
from torch.autograd import Variable
import numpy as np
from onmt.Utils import aeq, sequence_mask


class AttSRU(nn.Module):
    def __init__(self, input_size, attention_size, output_size,
                 dropout):
        super(AttSRU, self).__init__()
        self.linear_in = nn.Linear(input_size, 3 * output_size, bias=False)
        self.linear_hidden = nn.Linear(output_size, output_size, bias=False)
        self.linear_ctx = nn.Linear(output_size, output_size, bias=False)
        self.linear_enc = nn.Linear(output_size, output_size, bias=False)
        self.linear_attn_in = nn.Linear(output_size, output_size, bias=False)
        self.output_size = output_size
        self.attn = DotAttention(attention_size)
        self.dropout = nn.Dropout(dropout)
        self.preact_ln = LayerNorm(3 * output_size)
        self.enc_ln = LayerNorm(output_size)
        self.trans_h_ln = LayerNorm(output_size)
        self.trans_c_ln = LayerNorm(output_size)
        self.attn_in_ln = LayerNorm(output_size)

    def init_params(self):
        self.preact_ln.init_params()
        self.enc_ln.init_params()
        self.trans_h_ln.init_params()
        self.trans_c_ln.init_params()

    def forward(self, prev_layer, hidden, enc_output, memory_length):
        """
        :param prev_layer: targetL x batch x output_size
        :param hidden: batch x output_size
        :param enc_output: (targetL x batch) x sourceL x output_size
        :return:
        """
        # targetL x batch x output_size
        preact = self.linear_in(self.dropout(prev_layer))
        pctx = self.linear_enc(self.dropout(enc_output))
        preact = self.preact_ln(preact)
        pctx = self.enc_ln(pctx)

        z, h_gate, prev_layer_t = preact.split(self.output_size, dim=-1)
        z, h_gate = F.sigmoid(z), F.sigmoid(h_gate)

        ss = []
        for i in range(prev_layer.size(0)):
            s = (1. - z[i]) * hidden + z[i] * prev_layer_t[i]
            # targetL x batch x output_size
            ss += [s.squeeze(0)]
            # batch x output_size
            hidden = s

        # (targetL x batch) x output_size
        ss = torch.stack(ss)
        attn_in = self.attn_in_ln(
            self.linear_attn_in(self.dropout(ss)))\
            .transpose(0, 1).contiguous()
        attn_out, p_attn = self.attn(attn_in, pctx, memory_length)
        attn_out = attn_out / np.sqrt(self.output_size)

        attn = {"std": p_attn}

        trans_h = self.linear_hidden(self.dropout(ss))
        trans_c = self.linear_ctx(self.dropout(attn_out))
        trans_h = self.trans_h_ln(trans_h)
        trans_c = self.trans_c_ln(trans_c)
        # trans_h, trans_c = F.tanh(trans_h), F.tanh(trans_c)
        out = trans_h + trans_c
        out = F.tanh(out)
        out = out.view(prev_layer.size())
        out = (1. - h_gate) * out + h_gate * prev_layer

        return out, hidden, attn


class BiSRU(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(BiSRU, self).__init__()
        self.input_linear = nn.Linear(input_size, 3 * output_size,
                                      bias=False)
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        self.preact_ln = LayerNorm(3 * output_size)

    def init_params(self):
        self.preact_ln.init_params()

    def forward(self, input):
        pre_act = self.input_linear(self.dropout(input))

        pre_act = self.preact_ln(pre_act)

        h_gate = pre_act[:, :, 2 * self.output_size:]
        g, x = pre_act[:, :, :2 * self.output_size]\
            .split(self.output_size, dim=-1)
        gf, gb = F.sigmoid(g).split(self.output_size // 2, dim=-1)
        x_f, x_b = x.split(self.output_size // 2, dim=-1)
        h_gate = F.sigmoid(h_gate)
        h_f_pre = gf * x_f
        h_b_pre = gb * x_b

        h_i_f = Variable(h_f_pre.data.new(gf[0].size()).zero_(),
                         requires_grad=False)
        h_i_b = Variable(h_f_pre.data.new(gf[0].size()).zero_(),
                         requires_grad=False)

        h_f, h_b = [], []
        for i in range(input.size(0)):
            h_i_f = (1. - gf[i]) * h_i_f + h_f_pre[i]
            h_i_b = (1. - gb[-(i + 1)]) * h_i_b + h_b_pre[-(i + 1)]
            h_f += [h_i_f]
            h_b += [h_i_b]

        h = torch.cat([torch.stack(h_f), torch.stack(h_b[::-1])], dim=-1)

        output = (1. - h_gate) * h + input * h_gate

        return output


class SRDecoderState(DecoderState):
    def __init__(self, hidden_size, rnnstate):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        self.hidden = rnnstate
        self.coverage = None

    @property
    def _all(self):
        return self.hidden

    def update_state(self, rnnstate):
        self.hidden = rnnstate

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars)


class DotAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.

    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = \sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`

    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """
    def __init__(self, dim):
        super(DotAttention, self).__init__()
        self.dim = dim
        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        h_s_ = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s_)

    def forward(self, input, memory_bank, memory_lengths=None, coverage=None):
        """
        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        batch, sourceL, dim = memory_bank.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)

        # compute attention scores, as in Luong et al.
        align = self.score(input, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths)
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.data.masked_fill_(1 - mask, -float('inf'))

        # Softmax to normalize attention weights
        align_vectors = self.sm(align.view(batch*targetL, sourceL)
                                / np.sqrt(dim_))
        align_vectors = align_vectors.view(batch, targetL, sourceL)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)

        if one_step:
            # Check output sizes
            c = c.squeeze(1)
            batch_, dim_ = c.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, sourceL_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        else:
            c = c.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

            # Check output sizes
            targetL_, batch_, sourceL_ = align_vectors.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        return c, align_vectors
