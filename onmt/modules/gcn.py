from __future__ import division
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout, edge_dropout, n_layers, activation, highway, bias=True):
        super(GraphConvolution, self).__init__()        
        self.dropout = nn.Dropout(dropout)
        self.in_features = in_features
        self.n_layers = n_layers
        self.out_features = out_features
        self.edge_dropout = edge_dropout
        self.activation = activation
        self.highway = highway
        self.layer_1 = GraphConvolutionLayer(in_features,
                out_features,
                edge_dropout,
                activation,
                highway,
                bias)
        if n_layers == 2:
            self.layer_2 = GraphConvolutionLayer(in_features,
                    out_features,
                    edge_dropout,
                    activation,
                    highway,
                    bias)
        else:
            self.layer_2 = None
        assert(n_layers == 1 or n_layers==2)
            
    def forward(self, inputs, adj):
        features = self.dropout(self.layer_1(inputs, adj))
        if self.layer_2:
            features = self.dropout(self.layer_2(features, adj))
        return features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
                + str(self.in_features) + ' -> ' \
                + str(self.out_features) + ', ' \
                + 'activation=' + str(self.activation) + ', ' \
                + 'highway=' + str(self.highway) + ', ' \
                + 'layers=' + str(self.n_layers) + ', ' \
                + 'dropout=' + str(self.dropout.p) + ', ' \
                + 'edge_dropout=' + str(self.edge_dropout) + ')'                

class GraphConvolutionLayer(nn.Module):
    """
    From https://github.com/tkipf/pygcn.
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, edge_dropout, activation, highway, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        out_features = int(out_features)
        self.out_features = out_features
        self.edge_dropout = nn.Dropout(edge_dropout)
        self.highway = highway
        self.activation = activation
        self.weight = Parameter(torch.Tensor(3, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(3, 1, out_features))
        if highway != "":
            assert(in_features == out_features)
            self.weight_highway = Parameter(torch.Tensor(in_features, out_features))
            self.bias_highway = Parameter(torch.Tensor(1, out_features))
        
        else:
            self.bias = None
            self.register_parameter('bias', None)         
            
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)           
            
    def forward(self, inputs, adj):
        features = self.edge_dropout(inputs)
        outputs = []
        for i in range(features.size()[1]):
            support = torch.bmm(
                features[:, i, :].unsqueeze(0).expand(self.weight.size(0), *features[:, i, :].size()), 
                self.weight
            )
            if self.bias is not None:
                support += self.bias.expand_as(support)

            output = torch.mm(
                adj[:, i, :].transpose(1,2).contiguous().view(support.size(0) * support.size(1), -1).transpose(0, 1), 
                support.view(support.size(0) * support.size(1), -1)
            )
            outputs.append(output)
        if self.activation == "leaky_relu":
            output = F.leaky_relu(torch.stack(outputs, 1))
        elif self.activation == "relu":
            output = F.relu(torch.stack(outputs, 1))
        elif self.activation == "tanh":
            output = torch.tanh(torch.stack(outputs, 1))
        elif self.activation == "sigmoid":
            output = torch.sigmoid(torch.stack(outputs, 1))
        else:
            assert(False)

        if self.highway != "":
            transform = []
            for i in range(features.size()[1]):
                transform_batch = torch.mm(features[:, i, :], self.weight_highway)
                transform_batch += self.bias_highway.expand_as(transform_batch)
                transform.append(transform_batch)
            if self.highway == "leaky_relu":
                transform = F.leaky_relu(torch.stack(transform, 1))  
            elif self.highway == "relu":
                transform = F.relu(torch.stack(transform, 1))  
            elif self.highway == "tanh":
                transform = torch.tanh(torch.stack(transform, 1))
            elif self.highway == "sigmoid":
                transform = torch.sigmoid(torch.stack(transform, 1))
            else:
                assert(False)                
            carry = 1 - transform
            output = output * transform + features * carry
        return output