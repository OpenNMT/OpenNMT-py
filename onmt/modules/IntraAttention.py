import torch
import torch.nn as nn


class IntraAttention(nn.Module):
    """IntraAttention module as described in Paulus (2017) sect. (2)
    """

    def __init__(self, dim, bias=False, temporal=False):
        super(IntraAttention, self).__init__()
        self.dim = dim
        self.temporal = temporal
        self.linear = nn.Linear(dim, dim, bias=bias)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h_t, h, attn_history=None):
        """
        Args:
            h_t : [bs x dim]
            h   : [n x bs x dim]
            attn_history: None or [(t-1) x bs x n]
        Returns:
            C_t :  [bs x n]
            alpha: [bs x dim]
            attn_history: [t x bs x n]
        """
        bs, dim = h_t.size()
        n, _bs, _dim = h.size()
        assert (_bs, _dim) == (bs, dim)
        if attn_history is not None:
            _t, __bs, _n = attn_history.size()
            assert (__bs, _n) == (_bs, n)

        h_t = self.linear(h_t).unsqueeze(1)
        h = h.view(n, bs, dim)

        # e_t = [bs, 1, dim] bmm [bs, dim, n] = [bs, n] (after squeeze)
        e_t = h_t.bmm(h.transpose(0, 1).transpose(1, 2)).squeeze(1)

        next_attn_history = None
        alpha = None
        if self.temporal:
            if attn_history is None:
                next_attn_history = e_t.unsqueeze(0)
            else:
                # We substract the maximum value in order to ensure
                # numerical stability
                next_attn_history = torch.cat([attn_history,
                                               e_t.unsqueeze(0)], 0)
                m = next_attn_history.max(0)[0]
                e_t = (e_t - m).exp() / (attn_history - m).exp().sum(0)
                s_t = e_t.sum(1)
                alpha = e_t / s_t.unsqueeze(1)

        if alpha is None:
            alpha = self.softmax(e_t)

        c_t = alpha.unsqueeze(1).bmm(h.transpose(0, 1)).squeeze(1)

        if self.temporal:
            return c_t, alpha, next_attn_history
        return c_t, alpha
