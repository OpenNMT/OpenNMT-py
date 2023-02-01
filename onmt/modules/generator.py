""" Onmt NMT Model base class definition """
import torch.nn as nn

from onmt.modules.copy_generator import CopyGenerator


class Generator(nn.Module):
    def __init__(self, hid_sizes, gen_sizes,
                 shared=False, copy_attn=False, pad_idx=None):
        super(Generator, self).__init__()
        self.generators = nn.ModuleList()
        self.shared = shared
        self.hid_sizes = hid_sizes
        self.gen_sizes = gen_sizes

        def simple_generator(hid_size, gen_size):
            return nn.Linear(hid_size, gen_size)

        # First generator: next token prediction
        if copy_attn:
            self.generators.append(
                CopyGenerator(hid_sizes[0], gen_sizes[0], pad_idx))
        else:
            self.generators.append(
                simple_generator(hid_sizes[0], gen_sizes[0]))

        # Additional generators: target features
        for hid_size, gen_size in zip(hid_sizes[1:], gen_sizes[1:]):
            self.generators.append(
                simple_generator(hid_size, gen_size))

    def forward(self, dec_out):
        # if shared_decoder_embeddings, we slice the decoder output
        if self.shared:
            raise NotImplementedError()
            '''
            outs = []
            offset = 0
            for generator, s in zip(self.generators, self.rnn_sizes):
                sliced_dec_out = dec_out[:, offset:offset+s]
                out = generator(sliced_dec_out)
                offset += s
                outs.append(out)
            return outs
            '''
        else:
            return [generator(dec_out) for generator in self.generators]

    def __getitem__(self, i):
        return self.generators[0][i]
