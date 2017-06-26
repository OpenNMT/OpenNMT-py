
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            out = super(Bottle2, self).forward(input.view(size[0]*size[1], size[2], size[3]))
            return out.contiguous().view(size[0], size[1], size[2], size[3])

# class LayerNorm(nn.Module):
#     def __init__(self, d_hid, eps=1e-3):
#         super(LayerNorm, self).__init__()
#         self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)
#         self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
#         self.eps = eps
        
#     def forward(self, input):
        
#         mean = input.mean(1)
#         input_center = input - mean.expand_as(input)
        
#         var = input_center.pow(2).mean(1)
#         std_rep = (var.expand_as(input) + self.eps).sqrt()
#         output = self.b_2.unsqueeze(0).expand_as(input) + self.a_2.unsqueeze(0).expand_as(input) * (input_center / std_rep)
#         return output

#         # return (input_center / std_rep)

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
        mu = torch.mean(z, dim=1).unsqueeze(1)
        sigma = torch.std(z, dim=1).unsqueeze(1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
        return ln_out
            

# class LayerNorm(nn.Module):
#         def __init__(self, _):
#             super(LayerNorm, self).__init__()
                    
#         def forward(self, input,dummy=None):
#             if dummy is None:
#                 dummy = torch.zeros(input.size(0)).cuda()
#                 dummy_var = torch.ones(input.size(0)).cuda() # These may need to be Variables
#             x = input.transpose(0,1).contiguous()
#             x = F.batch_norm(x,running_mean=dummy,running_var=dummy,weight=None,bias=None,training=True, momentum=0.1,eps=1e-5)
#             return x.transpose(0,1)


class BottleLinear(Bottle, nn.Linear):
    pass
class BottleLayerNorm(Bottle, LayerNorm):
    pass
class BottleSoftmax(Bottle, nn.Softmax):
    pass
