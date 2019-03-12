import torch
import smatch.amr_edited as amrannot
import re
from collections import defaultdict

class AMRData:
    def __init__(self, words, traverse, graph_reent, graph_no_reent, with_reentrancies=False):
        self.idx = 0
        self.annotation = " ".join(words)
        self.traverse = traverse

        if len(self.traverse) == 0:
            self.parents = [-1]
        else:
            self.parents = [-1]*len(self.traverse)
        
        self.matrix = torch.IntTensor(3, len(self.traverse), len(self.traverse)).zero_()
        self.matrix[0, :, :] = torch.eye(len(self.traverse))
        num_edges = 0
        longest_dep = 0
        reentrancies = 0
        for edge_reent, edge_no_reent in zip(graph_reent, graph_no_reent):
            i, j = edge_reent
            longest_dep = max(longest_dep, j - i)
            i2, j2 = edge_no_reent
            assert(i == i2 or j == j2)
            if j != j2:
                reentrancies += 1
            self.parents[j2] = i2 + 1
            if not words[j2].startswith(":"):
                num_edges += 1
            if i == -1 or j == -1:
                continue
            if len(self.traverse) > 1:
                if with_reentrancies:  
                    self.matrix[1, i, j] = 1
                    self.matrix[2, j, i] = 1
                else:
                    self.matrix[1, i2, j2] = 1
                    self.matrix[2, j2, i2] = 1
        print(reentrancies, longest_dep)


    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self.annotation)
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.traverse)
    
    def __getitem__(self, key):
        return self.traverse[key]

    def __next__(self):
        self.idx += 1
        try:
            word = self.traverse[self.idx - 1]
            return word
        except IndexError:
            self.idx = 0
            raise StopIteration
    next = __next__
    
def parse(words, reentrancies):
    words2 = []
    for i, w in enumerate(words):
        if i > 0 and words[i] != ")" and words[i - 1] == ")" and not words[i].startswith(":"):
            words2.append(":dummy")
        words2.append(w)
    words = words2
    
    var2idx = {}
    idx2var = {}
    for i in range(len(words)):
        if i + 1 < len(words) and words[i + 1] == "/":
            if words[i] not in var2idx:
                idx2var[i + 2] = words[i]
                var2idx[words[i]] = i + 2

    i = 0
    lhs = [0]
    edges = []
    last_role = -1
    node = -2
    while i < len(words):
        if words[i].startswith(":") and words[i] != ":p":
            if words[i + 1] == "(":
                if not reentrancies:
                    b = i + 4
                else:
                    b = var2idx[words[i + 2]]
                edges.append((lhs[-1], i))
                edges.append((i, b))                
                lhs.append(b)
                i += 1
            elif i + 2 < len(words) and words[i + 2] == "/":
                if not reentrancies:
                    b = i + 3
                else:
                    b = var2idx[words[i + 1]]
                edges.append((lhs[-1], i))
                edges.append((i, b))
                i += 1
            else:
                edges.append((lhs[-1], i))
                edges.append((i, i + 1)) 
                i += 1
        elif words[i] == ")":
            lhs.pop()
            i += 1
        else:
            if words[i][0] != "(" and words[i] != "/" and (i + 1 >= len(words) or words[i + 1] != "/"):
                if node == i - 3 and lhs[-1] != i:
                    edges.append((lhs[-1], i))
                node = i            
            i += 1
            
    traverse = []
    cnt = 0
    idxmap = {-1: -1}
    for i in range(len(words)):
        if i + 1 < len(words) and words[i + 1] == "/":
            continue
        if words[i] == "/":
            continue
        idxmap[i] = cnt
        cnt += 1
        traverse.append(words[i])      
    edges_novar = []

    for e in edges:
        if e[0] in idxmap:
            e0 = idxmap[e[0]]
        else:
            e0 = idxmap[var2idx[words[e[0]]]]
        if e[1] in idxmap:
            e1 = idxmap[e[1]]
        else:
            e1 = idxmap[var2idx[words[e[1]]]]
        edges_novar.append((e0, e1))
    if edges_novar == []:
        edges_novar = [(-1, 0)]
    if traverse == []:
        traverse = words 
    if (-1, 0) not in edges_novar:
        edges_novar = [(-1, 0)] + edges_novar
    if (-1, 0) not in edges:
        edges = [(-1, 0)] + edges
    return traverse, edges_novar, edges

def extract_amr_features(line, with_reentrancies):
    global i
    if not line:
        return [], []
    words = tuple(line)
    traverse, graph_reent, _ = parse(line, True)
    _, graph_no_reent, _ = parse(line, False)
    return AMRData(words, traverse, graph_reent, graph_no_reent, with_reentrancies)
