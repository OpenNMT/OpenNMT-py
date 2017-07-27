import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt
import math


class Embeddings(nn.Module):
    def __init__(self, opt, dicts, feature_dicts=None):
        self.positional_encoding = opt.position_encoding
        if self.positional_encoding:
            self.pe = self.make_positional_encodings(opt.word_vec_size, 5000)
            if len(opt.gpus) > 0:
                self.pe = self.pe.cuda()

        self.word_vec_size = opt.word_vec_size

        super(Embeddings, self).__init__()
        self.word_lut = nn.Embedding(len(dicts),
                                     opt.word_vec_size,
                                     padding_idx=dicts.stoi[onmt.IO.PAD_WORD])
        # Word embeddings.
        self.dropout = nn.Dropout(p=opt.dropout)
        self.feature_dicts = feature_dicts
    
        # Feature embeddings.
        if self.feature_dicts:
            self.feature_luts = nn.ModuleList([
                nn.Embedding(len(feature_dict),
                             opt.feature_vec_size,
                             padding_idx=feature_dict.stoi[onmt.IO.PAD_WORD])
                for feature_dict in feature_dicts])

            # MLP on features and words.
            self.activation = nn.ReLU()
            self.linear = onmt.modules.BottleLinear(
                opt.word_vec_size +
                len(feature_dicts) * opt.feature_vec_size,
                opt.word_vec_size)
        else:
            self.feature_luts = nn.ModuleList([])

    def make_positional_encodings(self, dim, max_len):
        pe = torch.FloatTensor(max_len, 1, dim).fill_(0)
        for i in range(dim):
            for j in range(max_len):
                k = float(j) / (10000.0 ** (2.0*i / float(dim)))
                pe[j, 0, i] = math.cos(k) if i % 2 == 1 else math.sin(k)
        return pe

    def load_pretrained_vectors(self, emb_file):
        if emb_file is not None:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, src_input):
        """
        Embed the words or utilize features and MLP.

        Args:
            src_input (LongTensor): len x batch x nfeat

        Return:
            emb (FloatTensor): len x batch x input_size
        """
        word = self.word_lut(src_input[:, :, 0])
        emb = word
        if self.feature_dicts:
            features = [feature_lut(src_input[:, :, j+1])
                        for j, feature_lut in enumerate(self.feature_luts)]

            # Apply one MLP layer.
            emb = self.activation(
                self.linear(torch.cat([word] + features, -1)))
        if self.positional_encoding:
            emb = emb + Variable(self.pe[:emb.size(0), :1, :emb.size(2)]
                                 .expand_as(emb))
            emb = self.dropout(emb)
        return emb
