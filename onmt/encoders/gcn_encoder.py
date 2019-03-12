"""Define RNN-based encoders."""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory

from onmt.modules.gcn import GraphConvolution

class GCNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False, gcn_dropout=0.0,
                 gcn_edge_dropout=0.0, n_gcn_layers=1, activation='',
                 highway=''):
                    
        super(GCNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        
        use_rnn = bidirectional
        self.rnn = None        
        if use_rnn:
            self.rnn, self.no_pack_padded_seq = \
                rnn_factory(rnn_type,
                            input_size=embeddings.embedding_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional)
            
        self.gcn = GraphConvolution(embeddings.embedding_size,
            embeddings.embedding_size,
            gcn_dropout,
            gcn_edge_dropout,
            n_gcn_layers,
            activation,
            highway)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    def forward(self, src, lengths=None):
        "See :obj:`EncoderBase.forward()`"
        
        if isinstance(src, tuple):
            src, adj, trees = src
            
        self._check_args(src, lengths)
        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        if self.rnn is not None:
            packed_emb = emb
            if lengths is not None and not self.no_pack_padded_seq:
                # Lengths data is wrapped inside a Tensor.
                lengths_list = lengths.view(-1).tolist()
                packed_emb = pack(emb, lengths_list)

            memory_bank, encoder_final = self.rnn(packed_emb)

            if lengths is not None and not self.no_pack_padded_seq:
                memory_bank = unpack(memory_bank)[0]
        else:
            memory_bank = emb
            
        memory_bank_gcn = self.gcn(memory_bank, adj.float().cuda())
        
        if self.rnn is not None:
            n = memory_bank_gcn.size(2) // 2
            h1 = memory_bank_gcn[-1].unsqueeze(0)
            batch_size = memory_bank_gcn.size(1)
            h = Variable(h1.data.new(2, batch_size, n).fill_(0.))
            h[0] = h1[:,:,0:n]
            h[1] = h1[:,:,n:n * 2]
            encoder_final = (h, h)
            memory_bank = memory_bank_gcn
        else:
            h, _ = torch.max(memory_bank_gcn, 0)
            h = h.unsqueeze(0)
            encoder_final = (h, h)
            memory_bank = memory_bank_gcn

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)

        return encoder_final, memory_bank, lengths

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs
