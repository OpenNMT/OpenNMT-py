""" Misc classes """
import torch
import torch.nn as nn


# At the moment this class is only used by embeddings.Embeddings look-up tables
class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    Inputs are a 3d Tensor whose last dimension is the same length
    as the list.
    Outputs are the result of applying modules to inputs elementwise.
    An optional merge parameter allows the outputs to be reduced to a
    single Tensor.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, inputs):
        inputs_ = [feat.squeeze(2) for feat in inputs.split(1, dim=2)]
        assert len(self) == len(inputs_)
        outputs = [f(x) for f, x in zip(self, inputs_)]
        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, 2)
        elif self.merge == 'sum':
            return sum(outputs)
        else:
            return outputs


class ApplyTemperature(nn.Module):
    """Divides logits by temperature."""

    def __init__(self, temp, *args):
        self.temp = temp
        super(ApplyTemperature, self).__init__(*args)

    def forward(self, x):
        if self.temp == 0.0:
            return torch.argmax(x, dim=1)
        else:
            return torch.div(x, self.temp)


class RestrictToTopK(nn.Module):
    """Zeroes out all logits except for the k largest."""

    def __init__(self, k, *args):
        self.k = k
        super(RestrictToTopK, self).__init__(*args)

    def forward(self, x):
        top_values, top_indices = torch.topk(x, self.k, dim=1)
        kth_best = top_values[:, -1].view([-1, 1])
        kth_best = kth_best.repeat([1, x.shape[1]])
        kth_best = kth_best.type(torch.cuda.FloatTensor)

        keep = torch.ge(x, kth_best).type(torch.cuda.FloatTensor)

        # Set all logits that are not in the top-k to -100.
        # This puts the probabilities close to 0.
        x = (keep * x) + ((1-keep) * -100)
        return x








