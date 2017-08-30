import torch
import torch.nn as nn
from onmt.modules import BottleLinear
from onmt.IO import PAD_WORD
from onmt.modules import aeq

class Embeddings(nn.Module):

    def __init__(
            self, dicts, feature_dicts,
            word_vec_size, pre_word_vecs, fix_word_vecs,
            feat_merge, feat_vec_size, feat_vec_exponent,
            position_encoding, gpus, dropout):

        super(Embeddings, self).__init__()

        self.positional_encoding = position_encoding
        if self.positional_encoding:
            self.pe = self.make_positional_encodings(word_vec_size, 5000)
            if len(gpus) > 0:
                self.pe.cuda()
            self.dropout = nn.Dropout(p=dropout)

        self.feat_merge = feat_merge

        vocab_sizes = [len(dicts)]
        emb_sizes = [word_vec_size]
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
                # apply a layer of mlp to get it down to the correct dim
                self.mlp = nn.Sequential(BottleLinear(
                                        sum(emb_sizes),
                                        word_vec_size),
                                        nn.ReLU())

        self.emb_luts = \
            nn.ModuleList([
                nn.Embedding(vocab, dim,
                             padding_idx=dicts.stoi[PAD_WORD])
                for vocab, dim in zip(vocab_sizes, emb_sizes)])
        if pre_word_vecs:
            self._load_pretrained_vectors(pre_word_vecs)
        if fix_word_vecs:
            self.word_lut.weight.requires_grad = False

    @property
    def word_lut(self):
        return self.emb_luts[0]

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

    def make_positional_encodings(self, dim, max_len):
        pe = torch.arange(0, max_len).unsqueeze(1).expand(max_len, dim)
        div_term = 1 / torch.pow(10000, torch.arange(0, dim * 2, 2) / dim)
        pe = pe * div_term.expand_as(pe)
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(1)

    def _load_pretrained_vectors(self, emb_file):
        pretrained = torch.load(emb_file)
        self.word_lut.weight.data.copy_(pretrained)

    def merge(self, features):
        if self.feat_merge == 'concat':
            return torch.cat(features, 2)
        elif self.feat_merge == 'sum':
            return sum(features)
        else:
            return self.mlp(torch.cat(features, 2))

    def forward(self, src_input):
        """
        Return the embeddings for words, and features if there are any.
        Args:
            src_input (LongTensor): len x batch x nfeat
        Return:
            emb (FloatTensor): len x batch x self.embedding_size
        """
        in_length, in_batch, nfeat = src_input.size()
        aeq(nfeat, len(self.emb_luts))

        if len(self.emb_luts) == 1:
            emb = self.word_lut(src_input.squeeze(2))
        else:
            feat_inputs = (feat.squeeze(2)
                           for feat in src_input.split(1, dim=2))
            features = [lut(feat)
                        for lut, feat in zip(self.emb_luts, feat_inputs)]
            emb = self.merge(features)

        if self.positional_encoding:
            emb = emb + Variable(self.pe[:emb.size(0), :1, :emb.size(2)]
                                 .expand_as(emb))
            emb = self.dropout(emb)

        out_length, out_batch, emb_size = emb.size()
        aeq(in_length, out_length)
        aeq(in_length, out_length)
        aeq(emb_size, self.embedding_size)

        return emb
