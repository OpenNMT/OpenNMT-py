import torch
import torch.nn as nn
from torch.autograd import Variable
from onmt.modules import BottleLinear
from onmt.modules import aeq


class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    Inputs are a 3d Variable whose last dimension is the same length
    as the list.
    Outputs are the result of applying modules to inputs elementwise.
    An optional merge parameter allows the outputs to be reduced to a
    single Variable.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, input):
        inputs = [feat.squeeze(2) for feat in input.split(1, dim=2)]
        assert len(self) == len(inputs)
        outputs = [f(x) for f, x in zip(self, inputs)]
        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, 2)
        elif self.merge == 'sum':
            return sum(outputs)
        else:
            return outputs


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.arange(0, max_len).unsqueeze(1).expand(max_len, dim)
        div_term = 1 / torch.pow(10000, torch.arange(0, dim * 2, 2) / dim)
        pe = pe * div_term.expand_as(pe)
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        self.pe = Variable(pe.unsqueeze(1))
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb):
        emb = emb + self.pe[:emb.size(0), :1, :emb.size(2)].expand_as(emb)
        emb = self.dropout(emb)
        return emb


class Embeddings(nn.Module):
    """
    Words embeddings dictionary for Encoder/Decoder.

    Args:
        embedding_dim (int): size of the dictionary of embeddings.
        position_encoding (bool): use a sin to mark relative words positions.
        feat_merge (string): merge action for the features embeddings:
                    concat, sum or mlp.
        feat_dim_exponent (float): when using '-feat_merge concat', feature
                    embedding size is N^feat_dim_exponent, where N is the
                    number of values of feature takes.
        feat_embedding_dim (int): embedding dimension for features when using
                    '-feat_merge mlp'
        dropout (float): dropout probablity.
        padding_idx (int): padding index in the embedding dictionary.
        num_word_embeddings (int): size of dictionary of embeddings for words.
        num_feat_embeddings ([int], optional): list of size of dictionary
                                    of embeddings for each feature.
    """
    def __init__(self, embedding_dim, position_encoding, feat_merge,
                 feat_vec_exponent, feat_vec_size, dropout, padding_idx,
                 feat_pads, word_vocab_size, feat_vocab_sizes=[]):
        super(Embeddings, self).__init__()

        self.padding_idx = padding_idx
        pad_indices = [padding_idx] + feat_pads

        if feat_merge == 'concat':
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]
        else:
            feat_dim = embedding_dim if feat_merge == 'sum' else feat_vec_size
            feat_dims = [feat_dim] * len(feat_vocab_sizes)
        vocab_sizes = [word_vocab_size] + feat_vocab_sizes
        emb_dims = [embedding_dim] + feat_dims

        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else embedding_dim)

        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [nn.Embedding(vocab, dim, padding_idx=pad)
                      for vocab, dim, pad in emb_params]
        emb_luts = Elementwise(feat_merge, embeddings)

        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)

        if feat_merge == 'mlp':
            in_dim = sum(emb_dims)
            out_dim = feat_vec_size
            mlp = nn.Sequential(BottleLinear(in_dim, out_dim), nn.ReLU())
            self.make_embedding.add_module('mlp', mlp)

        if position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module('pe', pe)

    @property
    def word_lut(self):
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file, fixed):
        if emb_file:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)
            if fixed:
                self.word_lut.weight.requires_grad = False

    def forward(self, input):
        """
        Return the embeddings for words, and features if there are any.
        Args:
            input (LongTensor): len x batch x nfeat
        Return:
            emb (FloatTensor): len x batch x self.embedding_size
        """
        in_length, in_batch, nfeat = input.size()
        aeq(nfeat, len(self.emb_luts))

        emb = self.make_embedding(input)

        out_length, out_batch, emb_size = emb.size()
        aeq(in_length, out_length)
        aeq(in_length, out_length)
        aeq(emb_size, self.embedding_size)

        return emb
