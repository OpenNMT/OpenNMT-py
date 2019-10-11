"""Define GGNN-based encoders."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory

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
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A, nodes):
        A_in = A[:, :, :nodes*self.n_edge_types]
        A_out = A[:, :, nodes*self.n_edge_types:]

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
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
       which is based on the paper "Gated Graph Sequence Neural Networks".
       FIXME: Add paper to refs.bib in docs/source.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [LSTM]
       annotation_dim (int) : FIXME add description
       state_dim (int) : FIXME
       n_edge_types (int) : FIXME
       n_node (int) : FIXME
       n_steps (int): FIXME
       src_vocab (int): FIXME add description
    """

    def __init__(self, rnn_type, annotation_dim, state_dim,
                 n_edge_types, n_node, n_steps, src_vocab):
        super(GGNNEncoder, self).__init__()

        assert (state_dim >= annotation_dim,  \
                'state_dim must be no less than annotation_dim')

        self.state_dim = state_dim
        self.annotation_dim = annotation_dim
        self.n_edge_types = n_edge_types
        self.n_node = n_node
        self.n_steps = n_steps

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = GGNNAttrProxy(self, "in_")
        self.out_fcs = GGNNAttrProxy(self, "out_")

        # Propogation Model
        self.propogator = GGNNPropogator(self.state_dim, self.n_node, self.n_edge_types)

        self._initialization()

        # Initialize the bridge layer
        self._initialize_bridge(rnn_type, self.state_dim, 1)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.annotation_dim,
            opt.state_dim,
            opt.n_edge_types,
            opt.n_node,
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
        # First length tells us how many nodes for this batch
        nodes = lengths[0].item()

        if False: # torch.cuda.is_available():
            print("SJKDEBUG: cuda nodes,lengths,self.state_dim",nodes,lengths,self.state_dim,flush=True)
            prop_state=torch.cuda.FloatTensor(src.size()[1],nodes,self.state_dim).fill_(0)
            A=torch.cuda.FloatTensor(src.size()[1],nodes,nodes*self.n_edge_types*2).fill_(0)
        else:
            print("SJKDEBUG: cpu nodes,lengths,self.state_dim",nodes,lengths,self.state_dim,flush=True)
            prop_state=torch.zeros(src.size()[1],nodes,self.state_dim)
            A=torch.zeros(src.size()[1],nodes,nodes*self.n_edge_types*2)
        for i in range(src.size()[1]):
            for j in range(len(src)):
                prop_state[i][j][src[j][i]] = 1.0
                if j > 0:
                  A[i][j-1][j] = 1.0
                  A[i][j][nodes + j-1] = 1.0

        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                print("SJKDEBUG: in_fcs, prop_state",self.in_fcs[i],prop_state,flush=True)
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, nodes*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, nodes*self.n_edge_types, self.state_dim)

            prop_state = self.propogator(in_states, out_states, prop_state, A, nodes)

        # If static annotation needed: join_state = torch.cat((prop_state, annotation), 2)
        prop_state = prop_state.transpose(0,1)
        # FIXME: for now, just average all nodes to get bridge input
        #    In future, may want to use start/end nodes if possible
        join_state = prop_state.mean(0)
        join_state = torch.stack((join_state,join_state,join_state,join_state))
        join_state = (join_state,join_state)

        print("SJKDEBUG:join_state",len(join_state),join_state[0].size())
        encoder_final = self._bridge(join_state)

        print("SJKDEBUG:encoder_final,prop_state,lengths",len(encoder_final),encoder_final[0].size(),prop_state.size(),lengths,flush=True)
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
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            print("SJKDEBUG:self.bridge[0],hidden.size(),self.total_hidden_dim",self.bridge[0],hidden.size(),self.total_hidden_dim,flush=True)
            outs = bottle_hidden(self.bridge[0], hidden)
            print("SJKDEBUG:outs.size()",outs.size())
        return outs

