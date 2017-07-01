

import torch
import torch.nn as nn


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

    The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

    """
    def __init__(self, dim, coverage=False):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, coverage=None):
        """
        input (FloatTensor): batch x dim
        context (FloatTensor): batch x sourceL x dim
        coverage (FloatTensor): batch x sourceL
        """
        # batch x dim x 1
        targetT = self.linear_in(input).unsqueeze(2)

        if coverage:
            context += self.linear_cover(coverage.view(-1).unsqueeze(1)) \
                           .view_as(context)
            context = self.tanh(context)

        # Get attention
        # batch x sourceL
        attn = torch.bmm(context, targetT).squeeze(2)

        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))

        attn = self.sm(attn)

        # Compute context.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL
        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        contextCombined = torch.cat((weightedContext, input), 1)

        final = self.linear_out(contextCombined)
        contextOutput = self.tanh(final)

        return contextOutput, attn
