from torch.autograd import Variable

def read_tree(line):
    parents = list(map(int, line))
    trees = dict()
    root = None
    for i in range(1, len(parents) + 1):
        if i - 1 not in trees.keys() and parents[i - 1] != -1:
            idx = i
            prev = None
            while True:
                parent = parents[idx - 1]
                if parent == -1:
                    patent = idx - 1
                tree = Tree()
                if prev is not None:
                    tree.add_child(prev)
                trees[idx - 1] = tree
                tree.idx = idx - 1
                if parent - 1 in trees.keys():
                    trees[parent - 1].add_child(tree)
                    break
                elif parent == 0:
                    root = tree
                    break
                else:
                    prev = tree
                    idx = parent
    return root

# def invert_rec(leaf, parent=None):
#     if leaf.parent is not None:
#         leaf.num_children += 1
#     leaf.children = leaf.parent
#     leaf.parent = parent
#     if leaf.children is not None:
#         invert_rec(leaf.children, leaf)
#     if parent is None:
#         return leaf
    
# tree object from stanfordnlp/treelstm
class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self._size = None

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)
        
    def hidden_traversal(self, out):
        out[self.idx] = self.state[1]
        for c in self.children:
            c.hidden_traversal(out)
        return out
    
    def topdown_hidden_traversal(self, out):
        out[self.idx] = self.topdown
        for c in self.children:
            c.topdown_hidden_traversal(out)
        return out
        
#     def init_states(self, init):
#         self.topdown_state = Variable(init)
#         for c in self.children:
#             c.init_states(init)
    
    def traversal(self):
        lst = [self.idx]
        for c in self.children:
            lst.extend(c.traversal())
        return lst
    
#     def traversal_trees(self):
#         lst = [self]
#         for c in self.children:
#             lst.extend(c.traversal_trees())
#         return lst    
    
#     def invert(self):
#         roots = []
#         for leaf in [n for n in self.traversal_trees() if n.num_children == 0]:
#             leaf = copy.deepcopy(leaf)
#             root = invert_rec(leaf, None)
#             if root.num_children > 0:
#                 roots.append(root)
#         return roots
        
    def size(self):
        if self._size is not None:
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size
    
    # def size(self):
    #     if getattr(self, '_size'):
    #         return self._size
    #     count = 1
    #     for i in range(self.num_children):
    #         count += self.children[i].size()
    #     self._size = count
    #     return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth