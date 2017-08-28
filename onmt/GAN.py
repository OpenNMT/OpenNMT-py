from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt
import onmt.modules
from onmt.modules import aeq

class CycleGAN(nn.Module):
    def __init__(self, G_A, G_B, D_A, D_B, \
                optimizer_D_A, optimizer_D_B, \
                optimizer_G, nll_loss, opt):
        self.G_A = G_A
        self.G_B = G_B
        self.D_A = D_A
        self.D_B = D_B
        self.optimizer_D_A = optimizer_D_A
        self.optimizer_D_B = optimizer_D_B
        self.optimizer_G = optimizer_G
        self.nll_loss = nll_loss
