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
            nn.LeakyReLU()
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
        nodes = self.n_node

        # FIXME: Perhaps this should be part of the model?
        if torch.cuda.is_available():
            prop_state=torch.cuda.FloatTensor(src.size()[1],nodes,self.state_dim).fill_(0)
            A=torch.cuda.FloatTensor(src.size()[1],nodes,nodes*self.n_edge_types*2).fill_(0)
        else:
            prop_state=torch.zeros(src.size()[1],nodes,self.state_dim)
            A=torch.zeros(src.size()[1],nodes,nodes*self.n_edge_types*2)

        # FIXME: We want the edges to be precomputed for full flexibility.
        # This probably means adding options to preprocess and somehow connecting
        # both an edge and bridge weight selection through to this encoder.
        if False:
            # Initialize edges as simple list, not interesting
            for i in range(src.size()[1]):
                for j in range(len(src)):
                    prop_state[i][j][src[j][i]] = 1.0
                    if j > 0:
                      A[i][j-1][j] = 1.0
                      A[i][j][nodes + j-1] = 1.0
        elif True:
            # Hardcoded tree for linear algebra equivalence explorations
            for i in range(src.size()[1]):
                n2p={}
                n2pidx=0
                nullval=src[src.size()[0]-1][i]
                dhval=57
                dxval=58
                # print("SJKDEBUG src 0,1:",src[0][i],src[1][i])
                for j in range(src.size()[0]):
                    # Initialize node state
                    # FIXME: There is probably a way to have an embedding layer here
                    if (src[j][i] != nullval):
                        n2p[j] = n2pidx
                        n2pidx += 1
                        prop_state[i][n2p[j]][src[j][i]] = 1.0
                if (n2pidx >= nodes-1):
                    print("Error: Too many nodes in graph: ",n2pidx," >= ",nodes-1,flush=True)
                    raise ValueError("Too many nodes in graph!")
                # print("SJKDEBUG i",i,"n2pidx",n2pidx,"nodes",nodes,"src.size()",src.size())
                # Last node is flagged as aggregator and left and right children are the programs
                prop_state[i][nodes-1][self.state_dim-1] = 1.0
                # For lin alg eq, edge types are 8 (left, right, LL, LR, RL, RR, 
                # startprog, endprog) and we do bidirectional as 2nd half of A matrix
                L_in = 0
                R_in = nodes
                LL_in = nodes*2
                LR_in = nodes*3
                RL_in = nodes*4
                RR_in = nodes*5
                LO_in = nodes*6
                Strt_in = nodes*7
                End_in = nodes*8
                if self.n_edge_types > 9:
                    LoopBody_in = nodes*9
                    LoopVar_in = nodes*10
                    L_out = nodes*11
                    R_out = nodes*12
                    LL_out = nodes*13
                    LR_out = nodes*14
                    RL_out = nodes*15
                    RR_out = nodes*16
                    LO_out = nodes*17
                    Strt_out = nodes*18
                    End_out = nodes*19
                    LoopBody_out = nodes*20
                    LoopVar_out = nodes*21
                else:
                    LoopBody_in = nodes*123
                    LoopVar_in = nodes*123
                    L_out = nodes*9
                    R_out = nodes*10
                    LL_out = nodes*11
                    LR_out = nodes*12
                    RL_out = nodes*13
                    RR_out = nodes*14
                    LO_out = nodes*15
                    Strt_out = nodes*16
                    End_out = nodes*17
                    LoopBody_out = -2
                    LoopVar_out = -2
                # To help in learning, label nodes with tree location data
                tree = 120 
                A[i][nodes-1][Strt_in+n2p[0]] = 1.0
                A[i][n2p[0]][Strt_out + nodes-1] = 1.0
                A[i][nodes-1][End_in + n2p[1]] = 1.0
                A[i][n2p[1]][End_out + nodes-1] = 1.0

                # Loop for both programs
                for lvl0 in range(2):
                    # Root is lvl0 children at +2 and +64
                    # Check that children aren't Null (max of 126 entries means [127] is always Null value)
                    prop_state[i][n2p[lvl0]][lvl0+tree] = 1.0
                    if (src[lvl0][i].item() == dhval or src[lvl0][i].item() == dxval):
                        A[i][n2p[lvl0]][LoopBody_in+n2p[lvl0+64]] = 1.0
                        A[i][n2p[lvl0+64]][LoopBody_out + n2p[lvl0]] = 1.0
                        loopvar = src[lvl0+2][i]
                        for varsearch in range(lvl0,nodes,2):
                            if (src[varsearch][i] == loopvar):
                                A[i][n2p[varsearch]][LoopVar_in+n2p[lvl0]] = 1.0
                                A[i][n2p[lvl0]][LoopVar_out + n2p[varsearch]] = 1.0
                    elif (src[lvl0+2][i] != nullval):
                        if (src[lvl0+64][i] != nullval):
                            A[i][n2p[lvl0]][L_in+n2p[lvl0+2]] = 1.0
                            A[i][n2p[lvl0+2]][L_out+ n2p[lvl0]] = 1.0
                            A[i][n2p[lvl0]][R_in+n2p[lvl0+64]] = 1.0
                            A[i][n2p[lvl0+64]][R_out + n2p[lvl0]] = 1.0
                        else:
                            A[i][n2p[lvl0]][LO_in+n2p[lvl0+2]] = 1.0
                            A[i][n2p[lvl0+2]][LO_out+ n2p[lvl0]] = 1.0
                    if (src[lvl0+4][i] != nullval):
                        A[i][n2p[lvl0]][LL_in+n2p[lvl0+4]] = 1.0
                        A[i][n2p[lvl0+4]][LL_out+ n2p[lvl0]] = 1.0
                    if (src[lvl0+66][i] != nullval):
                        A[i][n2p[lvl0]][RL_in+n2p[lvl0+66]] = 1.0
                        A[i][n2p[lvl0+66]][RL_out + n2p[lvl0]] = 1.0
                    if (src[lvl0+34][i] != nullval):
                        A[i][n2p[lvl0]][LR_in+n2p[lvl0+34]] = 1.0
                        A[i][n2p[lvl0+34]][LR_out+ n2p[lvl0]] = 1.0
                    if (src[lvl0+96][i] != nullval):
                        A[i][n2p[lvl0]][RR_in+n2p[lvl0+96]] = 1.0
                        A[i][n2p[lvl0+96]][RR_out + n2p[lvl0]] = 1.0
                    for lvl1 in (lvl0+2,lvl0+64):
                        # lvl1 is this node, children at +2 and +32
                        if (src[lvl1][i] == nullval):
                            continue
                        prop_state[i][n2p[lvl1]][lvl0+2+tree] = 1.0
                        if (src[lvl1+2][i] != nullval):
                            if (src[lvl1+32][i] != nullval):
                                A[i][n2p[lvl1]][L_in + n2p[lvl1+2]] = 1.0
                                A[i][n2p[lvl1+2]][L_out + n2p[lvl1]] = 1.0
                                A[i][n2p[lvl1]][R_in+n2p[lvl1+32]] = 1.0
                                A[i][n2p[lvl1+32]][R_out + n2p[lvl1]] = 1.0
                            else:
                                A[i][n2p[lvl1]][LO_in + n2p[lvl1+2]] = 1.0
                                A[i][n2p[lvl1+2]][LO_out + n2p[lvl1]] = 1.0
                        elif (src[lvl1+32][i] != nullval):
                            print("Error: Found right node without left node!",flush=True)
                            raise ValueError("Found right node without left node!")
                        if (src[lvl1+4][i] != nullval):
                            A[i][n2p[lvl1]][LL_in + n2p[lvl1+4]] = 1.0
                            A[i][n2p[lvl1+4]][LL_out + n2p[lvl1]] = 1.0
                        if (src[lvl1+34][i] != nullval):
                            A[i][n2p[lvl1]][RL_in+n2p[lvl1+34]] = 1.0
                            A[i][n2p[lvl1+34]][RL_out + n2p[lvl1]] = 1.0
                        if (src[lvl1+18][i] != nullval):
                            A[i][n2p[lvl1]][LR_in + n2p[lvl1+18]] = 1.0
                            A[i][n2p[lvl1+18]][LR_out + n2p[lvl1]] = 1.0
                        if (src[lvl1+48][i] != nullval):
                            A[i][n2p[lvl1]][RR_in+n2p[lvl1+48]] = 1.0
                            A[i][n2p[lvl1+48]][RR_out + n2p[lvl1]] = 1.0
                        for lvl2 in (lvl1+2,lvl1+32):
                            if (src[lvl2][i] == nullval):
                                continue
                            prop_state[i][n2p[lvl2]][lvl0+4+tree] = 1.0
                            if (src[lvl2+2][i] != nullval):
                                if (src[lvl2+16][i] != nullval):
                                    A[i][n2p[lvl2]][L_in + n2p[lvl2+2]] = 1.0
                                    A[i][n2p[lvl2+2]][L_out + n2p[lvl2]] = 1.0
                                    A[i][n2p[lvl2]][R_in+n2p[lvl2+16]] = 1.0
                                    A[i][n2p[lvl2+16]][R_out + n2p[lvl2]] = 1.0
                                else:
                                    A[i][n2p[lvl2]][LO_in + n2p[lvl2+2]] = 1.0
                                    A[i][n2p[lvl2+2]][LO_out + n2p[lvl2]] = 1.0
                            if (src[lvl2+4][i] != nullval):
                                A[i][n2p[lvl2]][LL_in + n2p[lvl2+4]] = 1.0
                                A[i][n2p[lvl2+4]][LL_out + n2p[lvl2]] = 1.0
                            if (src[lvl2+18][i] != nullval):
                                A[i][n2p[lvl2]][RL_in+n2p[lvl2+18]] = 1.0
                                A[i][n2p[lvl2+18]][RL_out + n2p[lvl2]] = 1.0
                            if (src[lvl2+10][i] != nullval):
                                A[i][n2p[lvl2]][LR_in + n2p[lvl2+10]] = 1.0
                                A[i][n2p[lvl2+10]][LR_out + n2p[lvl2]] = 1.0
                            if (src[lvl2+24][i] != nullval):
                                A[i][n2p[lvl2]][RR_in+n2p[lvl2+24]] = 1.0
                                A[i][n2p[lvl2+24]][RR_out + n2p[lvl2]] = 1.0
                            for lvl3 in (lvl2+2,lvl2+16):
                                if (src[lvl3][i] == nullval):
                                    continue
                                prop_state[i][n2p[lvl3]][lvl0+6+tree] = 1.0
                                if (src[lvl3+2][i] != nullval):
                                    if (src[lvl3+8][i] != nullval):
                                        A[i][n2p[lvl3]][L_in + n2p[lvl3+2]] = 1.0
                                        A[i][n2p[lvl3+2]][L_out + n2p[lvl3]] = 1.0
                                        A[i][n2p[lvl3]][R_in+n2p[lvl3+8]] = 1.0
                                        A[i][n2p[lvl3+8]][R_out + n2p[lvl3]] = 1.0
                                    else:
                                        A[i][n2p[lvl3]][LO_in + n2p[lvl3+2]] = 1.0
                                        A[i][n2p[lvl3+2]][LO_out + n2p[lvl3]] = 1.0
                                if (src[lvl3+4][i] != nullval):
                                    A[i][n2p[lvl3]][LL_in + n2p[lvl3+4]] = 1.0
                                    A[i][n2p[lvl3+4]][LL_out + n2p[lvl3]] = 1.0
                                if (src[lvl3+10][i] != nullval):
                                    A[i][n2p[lvl3]][RL_in+n2p[lvl3+10]] = 1.0
                                    A[i][n2p[lvl3+10]][RL_out + n2p[lvl3]] = 1.0
                                if (src[lvl3+6][i] != nullval):
                                    A[i][n2p[lvl3]][LR_in + n2p[lvl3+6]] = 1.0
                                    A[i][n2p[lvl3+6]][LR_out + n2p[lvl3]] = 1.0
                                if (src[lvl3+12][i] != nullval):
                                    A[i][n2p[lvl3]][RR_in+n2p[lvl3+12]] = 1.0
                                    A[i][n2p[lvl3+12]][RR_out + n2p[lvl3]] = 1.0
                                for lvl4 in (lvl3+2,lvl3+8):
                                    if (src[lvl4][i] == nullval):
                                        continue
                                    prop_state[i][n2p[lvl4]][lvl0+8+tree] = 1.0
                                    if (src[lvl4+2][i] != nullval):
                                        prop_state[i][n2p[lvl4+2]][lvl0+10+tree] = 1.0
                                        if (src[lvl4+4][i] != nullval):
                                            prop_state[i][n2p[lvl4+4]][lvl0+10+tree] = 1.0
                                            A[i][n2p[lvl4]][L_in + n2p[lvl4+2]] = 1.0
                                            A[i][n2p[lvl4+2]][L_out + n2p[lvl4]] = 1.0
                                            A[i][n2p[lvl4]][R_in+n2p[lvl4+4]] = 1.0
                                            A[i][n2p[lvl4+4]][R_out + n2p[lvl4]] = 1.0
                                        else:
                                            A[i][n2p[lvl4]][LO_in + n2p[lvl4+2]] = 1.0
                                            A[i][n2p[lvl4+2]][LO_out + n2p[lvl4]] = 1.0

        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                # print("SJKDEBUG: in_fcs, prop_state",self.in_fcs[i],prop_state,flush=True)
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, nodes*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, nodes*self.n_edge_types, self.state_dim)

            prop_state = self.propogator(in_states, out_states, prop_state, A, nodes)

        # If static annotation needed: join_state = torch.cat((prop_state, annotation), 2)
        prop_state = prop_state.transpose(0,1)
        if False:
            # FIXME: for now, just average all nodes to get bridge input
            #    In future, may want to use start/end nodes if possible
            join_state = prop_state.mean(0)
            join_state = torch.stack((join_state,join_state,join_state,join_state))
        elif True:
            # Use roots of the 2 program trees as inputs to bridge
            # print("SJKDEBUG propstate size",prop_state.size())
            join_state = torch.stack((prop_state[nodes-1],prop_state[nodes-1],prop_state[nodes-1],prop_state[nodes-1]))
        join_state = (join_state,join_state)

        # print("SJKDEBUG:join_state",len(join_state),join_state[0].size())
        encoder_final = self._bridge(join_state)

        # print("SJKDEBUG:encoder_final,prop_state,lengths",len(encoder_final),encoder_final[0].size(),prop_state.size(),lengths,flush=True)
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
            # print("SJKDEBUG:self.bridge[0],hidden.size(),self.total_hidden_dim",self.bridge[0],hidden.size(),self.total_hidden_dim,flush=True)
            outs = bottle_hidden(self.bridge[0], hidden)
            # print("SJKDEBUG:outs.size()",outs.size())
        return outs

