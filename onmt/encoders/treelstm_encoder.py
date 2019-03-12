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

from onmt.modules.treelstm import ChildSumTreeLSTM, TopDownTreeLSTM

class TreeLSTMEncoder(EncoderBase):
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
                 use_bridge=False, bidir_treelstm=False):
        super(TreeLSTMEncoder, self).__init__()
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
        
        self.bidir_treelstm = bidir_treelstm
        if self.bidir_treelstm:
            self.childsumtreelstm = ChildSumTreeLSTM(
                embeddings.embedding_size,
                embeddings.embedding_size//2)

            self.topdown = TopDownTreeLSTM(
                embeddings.embedding_size//2,
                embeddings.embedding_size//2)
        else:
            self.childsumtreelstm = ChildSumTreeLSTM(
                embeddings.embedding_size,
                embeddings.embedding_size)                

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
        s_len, batch, emb_dim = emb.size()
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
        states = []
        contexts = []
        hiddens = []
        for i in range(batch):
            state, hidden, context = self.childsumtreelstm(
                trees[i],
                memory_bank[:, i, :]
            )
            if self.bidir_treelstm:
                state_down, hidden_down, context_down = self.topdown(
                    trees[i],
                    context,
                    state
                )
                context = torch.cat([context, context_down], 1)
                state = torch.cat([state, state_down], 1)
                hidden = torch.cat([hidden, hidden_down], 1)
            contexts.append(context.unsqueeze(1))
            hiddens.append(hidden)
            states.append(state)
        state_batch = torch.cat(states, 0)
        hidden_batch = torch.cat(hiddens, 0)
        context_batch = torch.cat(contexts, 1)

        if self.rnn is not None:
            n = context_batch.size(2) // 2
            hidden_batch = hidden_batch.unsqueeze(0)
            h = Variable(hidden_batch.data.new(2, batch, n).fill_(0.))
            h[0] = hidden_batch[:,:,0:n]
            h[1] = hidden_batch[:,:,n:n * 2]      

            state_batch = state_batch.unsqueeze(0)
            s = Variable(hidden_batch.data.new(2, batch, n).fill_(0.))
            s[0] = state_batch[:,:,0:n]
            s[1] = state_batch[:,:,n:n * 2]
            encoder_final = (h, s)
            memory_bank = context_batch

        else:
            encoder_final = (hidden_batch.unsqueeze(0), state_batch.unsqueeze(0)) 
            memory_bank = context_batch            

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
