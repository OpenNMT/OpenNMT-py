""" Onmt NMT Model base class definition """
import torch.nn as nn

from onmt.modules.copy_generator import CopyGenerator


class Generator(nn.Module):

    def __init__(self, hid_sizes, gen_sizes,
                 shared=False, copy_attn=False, pad_idx=None):
        super(Generator, self).__init__()
        self.feats_generators = nn.ModuleList()
        self.shared = shared
        self.hid_sizes = hid_sizes
        self.gen_sizes = gen_sizes

        def simple_generator(hid_size, gen_size):
            return nn.Linear(hid_size, gen_size)

        # First generator: next token prediction
        if copy_attn:
            self.tgt_generator = \
                CopyGenerator(hid_sizes[0], gen_sizes[0], pad_idx)
        else:
            self.tgt_generator = \
                simple_generator(hid_sizes[0], gen_sizes[0])

        # Additional generators: target features
        for hid_size, gen_size in zip(hid_sizes[1:], gen_sizes[1:]):
            self.feats_generators.append(
                simple_generator(hid_size, gen_size))

    def forward(self, dec_out, *args):
        scores = self.tgt_generator(dec_out, *args)

        feats_scores = []
        for generator in self.feats_generators:
            feats_scores.append(generator(dec_out))

        return scores, feats_scores
