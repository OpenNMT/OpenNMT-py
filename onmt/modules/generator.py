""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn

from torch.nn.modules.module import _addindent

from onmt.modules.util_class import Cast


class Generator(nn.Module):
    def __init__(self, rnn_size, sizes, gen_func):
        super(Generator, self).__init__()
        self.generators = nn.ModuleList()
        for i, size in enumerate(sizes):
            self.generators.append(
                nn.Sequential(
            nn.Linear(rnn_size,
                      size),
            Cast(torch.float32),
            gen_func
        ))

    def forward(self, dec_out):
        return [generator(dec_out) for generator in self.generators]

    def __getitem__(self, i):
        return self.generators[0][i]
