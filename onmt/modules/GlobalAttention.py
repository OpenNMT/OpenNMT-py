import torch
import torch.nn as nn
from onmt.modules.Util import BottleLinear
from onmt.modules import aeq


class GlobalAttention(nn.Module):
    """
    Luong Attention.

    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.


        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                      a

    Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.

    Luong Attention (dot, general):
    The full function is
    $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.

    * dot: $$score(h_t,{\overline{h}}_s) = h_t^T{\overline{h}}_s$$
    * general: $$score(h_t,{\overline{h}}_s) = h_t^T W_a {\overline{h}}_s$$

    Bahdanau Attention (mlp):
    $$c = \sum_{j=1}^{SeqLength}\a_jh_j$$.
    The Alignment-function $$a$$ computes an alignment as:
    $$a_j = softmax(v_a^T \tanh(W_a q + U_a h_j) )$$.

    """
    def __init__(self, dim, coverage=False, attn_type="dot"):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
                "Please select a valid attention type.")

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = BottleLinear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = BottleLinear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim*2, dim, bias=out_bias)

        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()
        self.mask = None

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def applyMask(self, mask):
        self.mask = mask

    def score(self, h_t, h_s):
        """
        h_t (FloatTensor): batch x dim
        h_s (FloatTensor): batch x src_len x dim
        returns scores (FloatTensor): batch x src_len:
            raw attention scores for each src index
        """

        # Check input sizes
        src_batch, _, src_dim = h_s.size()
        tgt_batch, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t = self.linear_in(h_t)
            return torch.bmm(h_s, h_t.unsqueeze(2)).squeeze(2)
        else:
            # MLP
            # batch x 1 x dim
            wq = self.linear_query(h_t).unsqueeze(1)
            # batch x src_len x dim
            uh = self.linear_context(h_s.contiguous())
            # batch x src_len x dim
            wquh = uh + wq.expand_as(uh)
            # batch x src_len x dim
            wquh = self.tanh(wquh)
            # batch x src_len
            return self.v(wquh.contiguous()).squeeze(2)

    def forward(self, input, context, coverage=None):
        """
        input (FloatTensor): batch x dim: decoder's rnn's output.
        context (FloatTensor): batch x src_len x dim: src hidden states
        coverage (FloatTensor): batch x src_len
        """

        # Check input sizes
        batch, sourceL, dim = context.size()
        batch_, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, sourceL_ = coverage.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        if self.mask is not None:
            beam_, batch_, sourceL_ = self.mask.size()
            aeq(batch, batch_*beam_)
            aeq(sourceL, sourceL_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            context += self.linear_cover(cover).view_as(context)
            context = self.tanh(context)

        # compute attention scores, as in Luong et al.
        a_t = self.score(input, context)

        if self.mask is not None:
            a_t.data.masked_fill_(self.mask, -float('inf'))

        # Softmax to normalize attention weights
        align_vector = self.sm(a_t)

        # the context vector c_t is the weighted average
        # over all the source hidden states
        c_t = torch.bmm(align_vector.unsqueeze(1), context).squeeze(1)

        # concatenate
        attn_h_t = self.linear_out(torch.cat([c_t, input], 1))
        if self.attn_type in ["general", "dot"]:
            attn_h_t = self.tanh(attn_h_t)

        # Check output sizes
        batch_, sourceL_ = align_vector.size()
        aeq(batch, batch_)
        aeq(sourceL, sourceL_)
        batch_, dim_ = attn_h_t.size()
        aeq(batch, batch_)
        aeq(dim, dim_)

        return attn_h_t, align_vector

    def score_all(self, h_t, h_s):
        """
        whole sequence version of score().
        h_t (FloatTensor): batch x tgt_len x dim
        h_s (FloatTensor): batch x src_len x dim
        returns scores (FloatTensor): batch x tgt_len x src_len:
            raw attention scores for each src index
        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            # TODO: MLP
            raise Exception("mlp not supported yet.")

    def forward_all(self, input, context, coverage=None):
        """
        whole sequence version of forward().
        input (FloatTensor): batch x tgt_len x dim: decoder's rnn's output.
        context (FloatTensor): batch x src_len x dim: src hidden states
        coverage (FloatTensor): None (not supported yet)
        """

        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)

        if self.mask is not None:
            beam_, batch_, sourceL_ = self.mask.size()
            aeq(batch, batch_*beam_)
            aeq(sourceL, sourceL_)

        # TODO: coverage
        if coverage is not None:
            raise Exception("coverage not supported yet.")

        # compute attention scores, as in Luong et al.
        align = self.score_all(input, context)

        if self.mask is not None:
            mask_ = self.mask.view(batch, 1, sourceL)  # make it broardcastable
            align.data.masked_fill_(mask_, -float('inf'))

        # Softmax to normalize attention weights
        align_vectors = self.sm(align.view(batch*targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, context)

        # concatenate
        concat_c = torch.cat([c, input], 2).view(batch*targetL, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)

        attn_h = attn_h.transpose(0, 1)
        align_vectors = align_vectors.transpose(0, 1)

        # Check output sizes
        targetL_, batch_, dim_ = attn_h.size()
        aeq(targetL, targetL_)
        aeq(batch, batch_)
        aeq(dim, dim_)
        targetL_, batch_, sourceL_ = align_vectors.size()
        aeq(targetL, targetL_)
        aeq(batch, batch_)
        aeq(sourceL, sourceL_)

        return attn_h, align_vectors
