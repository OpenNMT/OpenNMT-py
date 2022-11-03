""" Misc classes """
import torch
import torch.nn as nn


# At the moment this class is only used by embeddings.Embeddings look-up tables
class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    emb is a 3d Tensor whose last dimension is the same length
    as the list.
    emb_out is the result of applying modules to emb elementwise.
    An optional merge parameter allows the emb_out to be reduced to a
    single Tensor.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, emb):
        emb_ = [feat.squeeze(2) for feat in emb.split(1, dim=2)]
        assert len(self) == len(emb_)
        emb_out = [f(x) for f, x in zip(self, emb_)]
        if self.merge == 'first':
            return emb_out[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(emb_out, 2)
        elif self.merge == 'sum':
            return sum(emb_out)
        else:
            return emb_out


class Cast(nn.Module):
    """
    Basic layer that casts its emb to a specific data type. The same tensor
    is returned if the data type is already correct.
    """

    def __init__(self, dtype):
        super(Cast, self).__init__()
        self._dtype = dtype

    def forward(self, x):
        return x.to(self._dtype)
