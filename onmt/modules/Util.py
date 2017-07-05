
import torch
import torch.nn as nn


def aeq(*args):
    base = args[0]
    for a in args[1:]:
        assert a == base, str(args)


class Bottle(nn.Module):
        def forward(self, input):
            if len(input.size()) <= 2:
                return super(Bottle, self).forward(input)
            size = input.size()[:2]
            out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
            return out.contiguous().view(size[0], size[1], -1)


class Bottle2(nn.Module):
        def forward(self, input):
            if len(input.size()) <= 3:
                return super(Bottle2, self).forward(input)
            size = input.size()
            out = super(Bottle2, self).forward(input.view(size[0]*size[1],
                                                          size[2], size[3]))
            return out.contiguous().view(size[0], size[1], size[2], size[3])


class LayerNorm(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, dim=1)
        sigma = torch.std(z, dim=1)
        # HACK. PyTorch is changing behavior
        if mu.dim() == 1:
            mu = mu.unsqueeze(1)
            sigma = sigma.unsqueeze(1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out.mul(self.a_2.expand_as(ln_out)) \
            + self.b_2.expand_as(ln_out)
        return ln_out


class BottleLinear(Bottle, nn.Linear):
    pass


class BottleLayerNorm(Bottle, LayerNorm):
    pass


class BottleSoftmax(Bottle, nn.Softmax):
    pass
