import math
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    """
    A simple encoder convolutional -> recurrent neural network for
    audio input.

    Args:
        num_layers (int): number of encoder layers.
        bidirectional (bool): bidirectional encoder.
        rnn_size (int): size of hidden states of the rnn.
        dropout (float): dropout probablity.
        sample_rate (float): input spec
        window_size (int): input spec

    """
    def __init__(self, num_layers, bidirectional, rnn_size, dropout,
                 sample_rate, window_size):
        super(AudioEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = rnn_size

        self.layer1 = nn.Conv2d(1,   32, kernel_size=(41, 11),
                                padding=(0, 10), stride=(2, 2))
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32,  32, kernel_size=(21, 11),
                                padding=(0, 0), stride=(2, 1))
        self.batch_norm2 = nn.BatchNorm2d(32)

        input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        input_size = int(math.floor(input_size - 41) / 2 + 1)
        input_size = int(math.floor(input_size - 21) / 2 + 1)
        input_size *= 32
        self.rnn = nn.LSTM(input_size, rnn_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)

    def load_pretrained_vectors(self, opt):
        # Pass in needed options only when modify function definition.
        pass

    def forward(self, input, lengths=None):
        "See :obj:`onmt.modules.EncoderBase.forward()`"
        # (batch_size, 1, nfft, t)
        # layer 1
        input = self.batch_norm1(self.layer1(input[:, :, :, :]))

        # (batch_size, 32, nfft/2, t/2)
        input = F.hardtanh(input, 0, 20, inplace=True)

        # (batch_size, 32, nfft/2/2, t/2)
        # layer 2
        input = self.batch_norm2(self.layer2(input))

        # (batch_size, 32, nfft/2/2, t/2)
        input = F.hardtanh(input, 0, 20, inplace=True)

        batch_size = input.size(0)
        length = input.size(3)
        input = input.view(batch_size, -1, length)
        input = input.transpose(0, 2).transpose(1, 2)

        output, hidden = self.rnn(input)

        return hidden, output
