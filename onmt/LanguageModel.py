"""
This file defines the model architecture and the functionalities of a language model.
It re-uses the components of the decoder.
"""

import torch
import torch.nn as nn
import onmt
from onmt.Models import StackedLSTM, StackedGRU

class LM(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        input_size = opt.word_vec_size

        super(LM, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)

        stackedCell = StackedLSTM if opt.rnn_type == "LSTM" else StackedGRU
        self.rnn = stackedCell(opt.layers, input_size,
                               opt.rnn_size, opt.dropout)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden):
        emb = self.word_lut(input)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)

            output, hidden = self.rnn(emb_t, hidden)
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden