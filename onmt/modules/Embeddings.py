import torch
import torch.nn as nn
from torch.autograd import Variable
from onmt.modules import BottleLinear
from onmt.IO import PAD_WORD
from onmt.modules import aeq


class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    Inputs are one Variable per parameter.
    Outputs are the result of applying modules to inputs elementwise.
    An optional merge parameter allows
    """

    def __init__(self, *args, merge=None):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, *args):
        assert len(self) == len(args)
        outputs = [f(x) for f, x in zip(self, args)]
        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, 2)
        elif self.merge == 'sum':
            return sum(outputs)
        elif self.merge == 'mlp':
            return self.mlp(outputs)
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

    def __init__(
            self, dicts, feature_dicts,
            word_vec_size, pre_word_vecs, fix_word_vecs,
            feat_merge, feat_vec_size, feat_vec_exponent,
            position_encoding, gpus, dropout):

        super(Embeddings, self).__init__()

        self.feat_merge = feat_merge

        vocab_sizes = [len(dicts)]
        emb_sizes = [word_vec_size]
        '''
        if feature_dicts:
            vocab_sizes.extend(len(feat_dict) for feat_dict in feature_dicts)
            if feat_merge == 'concat':
                # Derive embedding sizes from each feature's vocab size
                emb_sizes.extend([int(feat_dict.size() ** feat_vec_exponent)
                                  for feat_dict in feature_dicts])
            elif feat_merge == 'sum':
                # All embeddings to be summed must be the same size
                emb_sizes.extend([word_vec_size] * len(feature_dicts))
            else:
                # mlp feature merge
                emb_sizes.extend([feat_vec_size] * len(feature_dicts))
        self.emb_luts = \
            nn.ModuleList([
                nn.Embedding(vocab, dim,
                             padding_idx=dicts.stoi[PAD_WORD])
                for vocab, dim in zip(vocab_sizes, emb_sizes)])
        if pre_word_vecs:
            self._load_pretrained_vectors(pre_word_vecs)
        if fix_word_vecs:
            self.word_lut.weight.requires_grad = False
        '''
        embeddings = [nn.Embedding(vocab,
                                   dim,
                                   padding_idx=dicts.stoi[PAD_WORD])
                      for vocab, dim in zip(vocab_sizes, emb_sizes)]
        emb_luts = Elementwise(embeddings, merge='first')

        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)

        if feat_merge == 'mlp':
            # note: not the same thing as self.embedding_size
            in_dim = sum(emb_lut.embedding_dim
                         for emb_lut in self.emb_luts.children())
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

    @property
    def embedding_size(self):
        """
        Returns sum of all feature dimensions if the merge action is concat.
        Otherwise, returns word vector size.
        """
        if self.feat_merge == 'concat':
            return sum(emb_lut.embedding_dim
                       for emb_lut in self.emb_luts.children())
        else:
            return self.word_lut.embedding_dim

    def _load_pretrained_vectors(self, emb_file):
        pretrained = torch.load(emb_file)
        self.word_lut.weight.data.copy_(pretrained)

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

        inputs = [feat.squeeze(2) for feat in input.split(1, dim=2)]
        emb = self.make_embedding(*inputs)

        out_length, out_batch, emb_size = emb.size()
        aeq(in_length, out_length)
        aeq(in_length, out_length)
        aeq(emb_size, self.embedding_size)

        return emb
