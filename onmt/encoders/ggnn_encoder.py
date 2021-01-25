"""Define GGNN-based encoders."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.encoders.encoder import EncoderBase


class GGNNAttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class GGNNPropogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(GGNNPropogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.LeakyReLU()
        )

    def forward(self, state_in, state_out, state_cur, edges, nodes):
        edges_in = edges[:, :, :nodes*self.n_edge_types]
        edges_out = edges[:, :, nodes*self.n_edge_types:]

        a_in = torch.bmm(edges_in, state_in)
        a_out = torch.bmm(edges_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNNEncoder(EncoderBase):
    """ A gated graph neural network configured as an encoder.
       Based on github.com/JamesChuanggg/ggnn.pytorch.git,
       which is based on the paper "Gated Graph Sequence Neural Networks"
       by Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [LSTM]
       src_ggnn_size (int) : Size of token-to-node embedding input
       src_word_vec_size (int) : Size of token-to-node embedding output
       state_dim (int) : Number of state dimensions in nodes
       n_edge_types (int) : Number of edge types
       bidir_edges (bool): True if reverse edges should be autocreated
       n_node (int) : Max nodes in graph
       bridge_extra_node (bool): True indicates only 1st extra node
          (after token listing) should be used for decoder init.
       n_steps (int): Steps to advance graph encoder for stabilization
       src_vocab (int): Path to source vocabulary.(The ggnn uses src_vocab
            during training because the graph is built using edge information
            which requires parsing the input sequence.)
    """

    def __init__(self, rnn_type, src_word_vec_size, src_ggnn_size,
                 state_dim, bidir_edges, n_edge_types, n_node,
                 bridge_extra_node, n_steps, src_vocab):
        super(GGNNEncoder, self).__init__()

        self.src_word_vec_size = src_word_vec_size
        self.src_ggnn_size = src_ggnn_size
        self.state_dim = state_dim
        self.n_edge_types = n_edge_types
        self.n_node = n_node
        self.n_steps = n_steps
        self.bidir_edges = bidir_edges
        self.bridge_extra_node = bridge_extra_node

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = GGNNAttrProxy(self, "in_")
        self.out_fcs = GGNNAttrProxy(self, "out_")

        # Find vocab data for tree builting
        f = open(src_vocab, "r")
        idx = 0
        self.COMMA = -1
        self.DELIMITER = -1
        self.idx2num = []
        found_n_minus_one = False
        for ln in f:
            ln = ln.strip('\n')
            ln = ln.split('\t')[0]
            if idx == 0 and ln != "<unk>":
                idx += 1
                self.idx2num.append(-1)
            if idx == 1 and ln != "<blank>":
                idx += 1
                self.idx2num.append(-1)
            if ln == ",":
                self.COMMA = idx
            if ln == "<EOT>":
                self.DELIMITER = idx
            if ln.isdigit():
                self.idx2num.append(int(ln))
                if int(ln) == n_node-1:
                    found_n_minus_one = True
            else:
                self.idx2num.append(-1)
            idx += 1

        assert self.COMMA >= 0, \
            "GGNN src_vocab must include ',' character"
        assert self.DELIMITER >= 0, \
            "GGNN src_vocab must include <EOT> token"
        assert found_n_minus_one, \
            "GGNN src_vocab must include node numbers for edge connections"

        # Propogation Model
        self.propogator = GGNNPropogator(self.state_dim, self.n_node,
                                         self.n_edge_types)

        self._initialization()

        # Initialize the bridge layer
        self._initialize_bridge(rnn_type, self.state_dim, 1)

        # Token embedding
        if src_ggnn_size > 0:
            self.embed = nn.Sequential(
                nn.Linear(src_ggnn_size, src_word_vec_size),
                nn.LeakyReLU()
            )
            assert self.src_ggnn_size >= self.DELIMITER, \
                "Embedding input must be larger than vocabulary"
            assert self.src_word_vec_size < self.state_dim, \
                "Embedding size must be smaller than state_dim"
        else:
            assert self.DELIMITER < self.state_dim, \
                "Vocabulary too large, consider -src_ggnn_size"

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.src_word_vec_size,
            opt.src_ggnn_size,
            opt.state_dim,
            opt.bidir_edges,
            opt.n_edge_types,
            opt.n_node,
            opt.bridge_extra_node,
            opt.n_steps,
            opt.src_vocab)

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        nodes = self.n_node
        batch_size = src.size()[1]
        first_extra = np.zeros(batch_size, dtype=np.int32)
        token_onehot = np.zeros((batch_size, nodes,
                                 self.src_ggnn_size if self.src_ggnn_size > 0
                                 else self.state_dim),
                                dtype=np.int32)
        edges = np.zeros((batch_size, nodes, nodes*self.n_edge_types*2),
                         dtype=np.int32)
        npsrc = src[:, :, 0].cpu().data.numpy().astype(np.int32)

        # Initialize graph using formatted input sequence
        for i in range(batch_size):
            tokens_done = False
            # Number of flagged nodes defines node count for this sample
            # (Nodes can have no flags on them, but must be in 'flags' list).
            flag_node = 0
            flags_done = False
            edge = 0
            source_node = -1
            for j in range(len(npsrc)):
                token = npsrc[j][i]
                if not tokens_done:
                    if token == self.DELIMITER:
                        tokens_done = True
                        first_extra[i] = j
                    else:
                        token_onehot[i][j][token] = 1
                elif token == self.DELIMITER:
                    flag_node += 1
                    flags_done = True
                    assert flag_node <= nodes, "Too many nodes with flags"
                elif not flags_done:
                    # The total number of integers in the vocab should allow
                    # for all features and edges to be defined.
                    if token == self.COMMA:
                        flag_node = 0
                    else:
                        num = self.idx2num[token]
                        if num >= 0:
                            token_onehot[i][flag_node][num+self.DELIMITER] = 1
                        flag_node += 1
                elif token == self.COMMA:
                    edge += 1
                    assert source_node == -1, \
                        f'Error in graph edge input: {source_node} unpaired'
                    assert edge < self.n_edge_types, \
                        "Too many edge types in input"
                else:
                    num = self.idx2num[token]
                    if source_node < 0:
                        source_node = num
                    else:
                        edges[i][source_node][num+nodes*edge] = 1
                        if self.bidir_edges:
                            edges[i][num][nodes*(edge+self.n_edge_types)
                                          + source_node] = 1
                        source_node = -1

        token_onehot = torch.from_numpy(token_onehot).float().to(src.device)
        if self.src_ggnn_size > 0:
            token_embed = self.embed(token_onehot)
            prop_state = torch.cat((token_embed, torch.zeros(
                (batch_size, nodes, self.state_dim - self.src_word_vec_size)
                 ).float().to(src.device)), 2)
        else:
            prop_state = token_onehot
        edges = torch.from_numpy(edges).float().to(src.device)

        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, nodes*self.n_edge_types,
                                       self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, nodes*self.n_edge_types,
                                         self.state_dim)

            prop_state = self.propogator(in_states, out_states, prop_state,
                                         edges, nodes)

        prop_state = prop_state.transpose(0, 1)
        if self.bridge_extra_node:
            # Use first extra node as only source for decoder init
            join_state = prop_state[first_extra, torch.arange(batch_size)]
        else:
            # Average all nodes to get bridge input
            join_state = prop_state.mean(0)
        join_state = torch.stack((join_state, join_state,
                                  join_state, join_state))
        join_state = (join_state, join_state)

        encoder_final = self._bridge(join_state)

        return encoder_final, prop_state, lengths

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
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.leaky_relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs
