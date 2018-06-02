import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from onmt.Models import rnn_factory

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
    def __init__(self, rnn_type, enc_layers, dec_layers, brnn,
                 enc_rnn_size, dec_rnn_size, enc_pooling, dropout,
                 sample_rate, window_size):
        super(AudioEncoder, self).__init__()
        self.enc_layers = enc_layers
        self.rnn_type = rnn_type
        self.dec_layers = dec_layers
        num_directions = 2 if brnn else 1
        self.num_directions = num_directions
        assert enc_rnn_size % num_directions == 0
        enc_rnn_size_real = enc_rnn_size // num_directions
        assert dec_rnn_size % num_directions == 0
        self.dec_rnn_size = dec_rnn_size
        dec_rnn_size_real = dec_rnn_size // num_directions
        self.dec_rnn_size_real = dec_rnn_size_real
        self.dec_rnn_size = dec_rnn_size
        input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        enc_pooling = enc_pooling.split(',')
        assert len(enc_pooling) == enc_layers or len(enc_pooling) == 1
        if len(enc_pooling) == 1:
            enc_pooling = enc_pooling * enc_layers
        enc_pooling = [int(p) for p in enc_pooling]
        self.enc_pooling = enc_pooling
        self.W = nn.Linear(enc_rnn_size, dec_rnn_size, bias=False)
        self.batchnorm_W = nn.BatchNorm1d(enc_rnn_size, affine=True)
        self.batchnorm_0 = nn.BatchNorm1d(input_size, affine=True)
        self.rnn_0, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=input_size,
                        hidden_size=enc_rnn_size_real,
                        num_layers=1,
                        dropout=dropout,
                        bidirectional=brnn)
        self.pool_0 = nn.MaxPool1d(enc_pooling[0])
        for l in range(enc_layers - 1):
            batchnorm = nn.BatchNorm1d(enc_rnn_size, affine=True)
            rnn, _ = \
                rnn_factory(rnn_type,
                            input_size=enc_rnn_size,
                            hidden_size=enc_rnn_size_real,
                            num_layers=1,
                            dropout=dropout,
                            bidirectional=brnn)
            setattr(self, 'rnn_%d'%(l+1), rnn)
            setattr(self, 'pool_%d'%(l+1), nn.MaxPool1d(enc_pooling[l+1]))
            setattr(self, 'batchnorm_%d'%(l+1), batchnorm)


    def load_pretrained_vectors(self, opt):
        # Pass in needed options only when modify function definition.
        pass

    def forward(self, input, lengths=None):
        "See :obj:`onmt.modules.EncoderBase.forward()`"
        # (batch_size, 1, nfft, t)
        # layer 1
        batch_size, _, nfft, t = input.size()
        input = input.transpose(0, 1).transpose(0, 3).contiguous().view(t, batch_size, nfft)
        #print (input.size())
        #print (lengths)
        orig_lengths = lengths
        lengths = lengths.view(-1).tolist()

        for l in range(self.enc_layers):
            rnn = getattr(self, 'rnn_%d'%l)
            pool = getattr(self, 'pool_%d'%l)
            batchnorm = getattr(self, 'batchnorm_%d'%l)
            stride = self.enc_pooling[l]
            t1, t2, t3 = input.size()
            input = batchnorm(input.contiguous().view(-1, t3)).contiguous().view(t1, t2, t3)
            packed_emb = pack(input, lengths)
            memory_bank, tmp = rnn(packed_emb)
            memory_bank = unpack(memory_bank)[0] # t, batch_size, nfft
            t, _, _ = memory_bank.size()
            #print (memory_bank.size())
            memory_bank = memory_bank.transpose(0,2) # nfft, batch_size, t
            memory_bank = pool(memory_bank) # nfft, batch_size, (t - pool)/2 + 1
            lengths = [int(math.floor((length - stride)/stride + 1)) for length in lengths]
            #print (t)
            assert memory_bank.size(2) == int(math.floor((t - stride) / stride + 1))
            memory_bank = memory_bank.transpose(0, 2) # t, batch_size, rnn_size
            #print (memory_bank.size())
            input = memory_bank

        memory_bank = memory_bank.contiguous().view(-1, memory_bank.size(2))
        memory_bank = self.batchnorm_W(memory_bank)
        memory_bank = self.W(memory_bank).view(-1, batch_size, self.dec_rnn_size)
        #memory_bank = memory_bank.view(-1, batch_size, self.dec_rnn_size)
        #print (memory_bank.size())
        #input = self.batch_norm1(self.layer1(input[:, :, :, :]))

        ## (batch_size, 32, nfft/2, t/2)
        #input = F.hardtanh(input, 0, 20, inplace=True)

        ## (batch_size, 32, nfft/2/2, t/2)
        ## layer 2
        #input = self.batch_norm2(self.layer2(input))

        ## (batch_size, 32, nfft/2/2, t/2)
        #input = F.hardtanh(input, 0, 20, inplace=True)

        #batch_size = input.size(0)
        #length = input.size(3)
        #input = input.view(batch_size, -1, length)
        #input = input.transpose(0, 2).transpose(1, 2)

        #sys.exit(1)

        #output, hidden = self.rnn(input)

        #return hidden, output
        state = memory_bank.new_full((self.dec_layers*self.num_directions, batch_size, 
                                      self.dec_rnn_size_real), 0)
        if self.rnn_type == 'LSTM':
            # The encoder hidden is  (layers*directions) x batch x dim.
            encoder_final = (state, state)
        else:
            encoder_final = state
        return encoder_final, memory_bank, orig_lengths.new_tensor(lengths)
