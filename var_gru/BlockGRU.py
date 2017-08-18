# encoding: utf-8
"""
 created by tesla on 8/16/17
"""
import torch
from torch.autograd import Variable
from torch.nn.modules.rnn import GRUCell
import numpy as np


class VarGRU(torch.nn.Module):
    """
    an recurrent neural network with GRU that produces variable length hidden size.
    use handcrafted for-loops instead of thnn backend.RNN or cuDNN RNN.
    sample a {0, 1} from a gumbel-softmax to determine how to apply a mask vector to hidden state vector.
    """

    def __init__(self, embed_dim, block_size=20, max_blocks=10, bi_directional=False, dropout=0, bias=True,
                 nonlinearty='tanh'):
        super(VarGRU).__init__()
        self.max_hidden_size = block_size * max_blocks
        self.embed_dim = embed_dim
        self.is_bi_directional = bi_directional
        self.dropout = dropout
        self.recurrent_unit = GRUCell(embed_dim, self.max_hidden_size, bias=bias)

        self.block_size = block_size
        self.max_blocks = max_blocks

    def gumbel_sample(self):
        pass

    def forward(self, input_, max_time_step, hx):
        """
        should base torch.nn._function.AutogradRNN
        input format should be (time, batch, embed_dim)

        at the very first, output hidden state have only 1 block features.
        :param input_:
        :param max_time_step:
        :return:
        """
        mask_list = np.concatenate((np.ones([self.block_size, ]), np.zeros([self.max_hidden_size - self.block_size, ])),
                                   axis=0)
        h0 = torch.mul(torch.FloatTensor(mask_list))
        flag = True
        output = []
        for i in range(max_time_step):
            hx = self.recurrent_unit(input_[i], hx)

            output.append(hx)
