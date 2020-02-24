""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn

from onmt.modules.util_class import Cast

from onmt.modules.copy_generator import CopyGenerator


class Generator(nn.Module):
    def __init__(self, rnn_sizes, gen_sizes, gen_func,
                 shared=False, copy_attn=False, pad_idx=None):
        super(Generator, self).__init__()
        self.generators = nn.ModuleList()
        self.shared = shared
        self.rnn_sizes = rnn_sizes
        self.gen_sizes = gen_sizes

        def simple_generator(rnn_size, gen_size, gen_func):
            return nn.Sequential(
                        nn.Linear(rnn_size, gen_size),
                        Cast(torch.float32),
                        gen_func)

        # create first generator
        if copy_attn:
            self.generators.append(
                CopyGenerator(rnn_sizes[0], gen_sizes[0], pad_idx))
        else:
            self.generators.append(
                simple_generator(rnn_sizes[0], gen_sizes[0], gen_func))

        # additional generators for features
        for rnn_size, gen_size in zip(rnn_sizes[1:], gen_sizes[1:]):
            self.generators.append(
                simple_generator(rnn_size, gen_size, gen_func))

    def forward(self, dec_out):
        # if shared_decoder_embeddings, we slice the decoder output
        if self.shared:
            outs = []
            offset = 0
            for generator, s in zip(self.generators, self.rnn_sizes):
                sliced_dec_out = dec_out[:, offset:offset+s]
                out = generator(sliced_dec_out)
                offset += s
                outs.append(out)
            return outs
        else:
            return [generator(dec_out) for generator in self.generators]

    def __getitem__(self, i):
        return self.generators[0][i]
