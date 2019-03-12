from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        
    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
                self.fh(child_h) +
                self.fx(inputs).repeat(len(child_h), 1)
            )
        fc = torch.mul(f, child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, tree, inputs, root=True):
        _ = [self.forward(tree.children[idx], inputs, False) for idx in range(tree.num_children)]
        
        if tree.num_children == 0:
            child_c = Variable(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_h = Variable(inputs[0].data.new(1, self.mem_dim).fill_(0.))
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
        tree.state = self.node_forward(inputs[tree.idx].unsqueeze(0), child_c, child_h)
        if root:
            hiddens = tree.hidden_traversal(
                Variable(inputs.data.new(inputs.size(0), self.mem_dim).fill_(0.))
            )
        else:
            hiddens = None

        return tree.state[0], tree.state[1], hiddens

    
class TopDownTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(TopDownTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.root = nn.Linear(in_dim, mem_dim)
        self.cell = nn.LSTMCell(mem_dim, mem_dim)
        
    def _forward(self, tree, inputs, state, root):
        if root:
            tree.topdown = torch.tanh(self.root(inputs[tree.idx].unsqueeze(0)))
            if state is None:
                tree.topdown_state = Variable(inputs.data.new(1, self.mem_dim).fill_(0.))
            else:
                tree.topdown_state = state
        else:
            tree.topdown, tree.topdown_state = self.cell(    
                inputs[tree.idx].unsqueeze(0),
                (tree.parent.topdown,
                tree.parent.topdown_state)
            )
            
        for child in tree.children:
            self._forward(child, inputs, state, False)
                        
    def forward(self, tree, inputs, state):   
        self._forward(tree, inputs, state, True)
        hiddens = tree.topdown_hidden_traversal(
            Variable(inputs.data.new(inputs.size(0), self.mem_dim).fill_(0.))
        )
        return tree.topdown_state, tree.topdown, hiddens
            
