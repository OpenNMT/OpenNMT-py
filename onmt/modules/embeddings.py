""" Embeddings module """
import math
import warnings

import torch
import torch.nn as nn

from onmt.modules.util_class import Elementwise
from onmt.utils.logging import logger


class SequenceTooLongError(Exception):
    pass


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim)
        step = step or 0
        if self.pe.size(0) < step + emb.size(0):
            raise SequenceTooLongError(
                f"Sequence is {emb.size(0) + step} but PositionalEncoding is"
                f" limited to {self.pe.size(0)}. See max_len argument."
            )
        emb = emb + self.pe[step:emb.size(0)+step]
        emb = self.dropout(emb)
        return emb


class Embeddings(nn.Module):
    """Words embeddings for encoder/decoder.

    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.


    .. mermaid::

       graph LR
          A[Input]
          C[Feature 1 Lookup]
          A-->B[Word Lookup]
          A-->C
          A-->D[Feature N Lookup]
          B-->E[MLP/Concat]
          C-->E
          D-->E
          E-->F[Output]

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feat_padding_idx (List[int]): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes (List[int], optional): list of size of dictionary
            of embeddings for each feature.
        position_encoding (bool): see :class:`~onmt.modules.PositionalEncoding`
        feat_merge (string): merge action for the features embeddings:
            concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
            embedding size is N^feat_dim_exponent, where N is the
            number of values the feature takes.
        feat_vec_size (int): embedding dimension for features when using
            `-feat_merge mlp`
        dropout (float): dropout probability.
        freeze_word_vecs (bool): freeze weights of word vectors.
    """

    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 position_encoding=False,
                 feat_merge="concat",
                 feat_vec_exponent=0.7,
                 feat_vec_size=-1,
                 feat_padding_idx=[],
                 feat_vocab_sizes=[],
                 dropout=0,
                 sparse=False,
                 freeze_word_vecs=False):
        self._validate_args(feat_merge, feat_vocab_sizes, feat_vec_exponent,
                            feat_vec_size, feat_padding_idx)

        if feat_padding_idx is None:
            feat_padding_idx = []
        self.word_padding_idx = word_padding_idx

        self.word_vec_size = word_vec_size

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse)
                      for vocab, dim, pad in emb_params]
        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else word_vec_size)

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)

        if feat_merge == 'mlp' and len(feat_vocab_sizes) > 0:
            in_dim = sum(emb_dims)
            mlp = nn.Sequential(nn.Linear(in_dim, word_vec_size), nn.ReLU())
            self.make_embedding.add_module('mlp', mlp)

        self.position_encoding = position_encoding

        if self.position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module('pe', pe)

        if freeze_word_vecs:
            self.word_lut.weight.requires_grad = False

    def _validate_args(self, feat_merge, feat_vocab_sizes, feat_vec_exponent,
                       feat_vec_size, feat_padding_idx):
        if feat_merge == "sum":
            # features must use word_vec_size
            if feat_vec_exponent != 0.7:
                warnings.warn("Merging with sum, but got non-default "
                              "feat_vec_exponent. It will be unused.")
            if feat_vec_size != -1:
                warnings.warn("Merging with sum, but got non-default "
                              "feat_vec_size. It will be unused.")
        elif feat_vec_size > 0:
            # features will use feat_vec_size
            if feat_vec_exponent != -1:
                warnings.warn("Not merging with sum and positive "
                              "feat_vec_size, but got non-default "
                              "feat_vec_exponent. It will be unused.")
        else:
            if feat_vec_exponent <= 0:
                raise ValueError("Using feat_vec_exponent to determine "
                                 "feature vec size, but got feat_vec_exponent "
                                 "less than or equal to 0.")
        n_feats = len(feat_vocab_sizes)
        if n_feats != len(feat_padding_idx):
            raise ValueError("Got unequal number of feat_vocab_sizes and "
                             "feat_padding_idx ({:d} != {:d})".format(
                                n_feats, len(feat_padding_idx)))

    @property
    def word_lut(self):
        """Word look-up table."""
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        """Embedding look-up table."""
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
        """

        if emb_file:
            pretrained = torch.load(emb_file)
            pretrained_vec_size = pretrained.size(1)
            if self.word_vec_size > pretrained_vec_size:
                self.word_lut.weight.data[:, :pretrained_vec_size] = pretrained
            elif self.word_vec_size < pretrained_vec_size:
                self.word_lut.weight.data \
                    .copy_(pretrained[:, :self.word_vec_size])
            else:
                self.word_lut.weight.data.copy_(pretrained)

    def forward(self, source, step=None):
        """Computes the embeddings for words and features.

        Args:
            source (LongTensor): index tensor ``(len, batch, nfeat)``

        Returns:
            FloatTensor: Word embeddings ``(len, batch, embedding_size)``
        """

        if self.position_encoding:
            for i, module in enumerate(self.make_embedding._modules.values()):
                if i == len(self.make_embedding._modules.values()) - 1:
                    source = module(source, step=step)
                else:
                    source = module(source)
        else:
            source = self.make_embedding(source)

        return source

    def update_dropout(self, dropout):
        if self.position_encoding:
            self._modules['make_embedding'][1].dropout.p = dropout


# Some utilitary functions for pretrained embeddings

def read_embeddings(path, skip_lines=0, filter_set=None):
    """
    Read an embeddings file in the glove format.
    """
    embs = dict()
    total_vectors_in_file = 0
    with open(path, 'rb') as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            if not line:
                break
            if len(line) == 0:
                # is this reachable?
                continue

            l_split = line.decode('utf8').strip().split(' ')
            if len(l_split) == 2:
                continue
            total_vectors_in_file += 1
            if filter_set is not None and l_split[0] not in filter_set:
                continue
            embs[l_split[0]] = [float(em) for em in l_split[1:]]
    return embs, total_vectors_in_file


def calc_vocab_load_stats(vocab, loaded_embed_dict):
    matching_count = len(
        set(vocab.stoi.keys()) & set(loaded_embed_dict.keys()))
    missing_count = len(vocab) - matching_count
    percent_matching = matching_count / len(vocab) * 100
    return matching_count, missing_count, percent_matching


def convert_to_torch_tensor(word_to_float_list_dict, vocab):
    dim = len(next(iter(word_to_float_list_dict.values())))
    tensor = torch.zeros((len(vocab), dim))
    for word, values in word_to_float_list_dict.items():
        tensor[vocab.stoi[word]] = torch.Tensor(values)
    return tensor


def prepare_pretrained_embeddings(opt, fields):
    if all([opt.both_embeddings is None,
            opt.src_embeddings is None,
            opt.tgt_embeddings is None]):
        return

    assert opt.save_data, "-save_data is required when using \
        pretrained embeddings."

    vocs = []
    for side in ['src', 'tgt']:
        try:
            vocab = fields[side].base_field.vocab
        except AttributeError:
            vocab = fields[side].vocab
        vocs.append(vocab)
    enc_vocab, dec_vocab = vocs

    skip_lines = 1 if opt.embeddings_type == "word2vec" else 0
    if opt.both_embeddings is not None:
        set_of_src_and_tgt_vocab = \
            set(enc_vocab.stoi.keys()) | set(dec_vocab.stoi.keys())
        logger.info("Reading encoder and decoder embeddings from {}".format(
            opt.both_embeddings))
        src_vectors, total_vec_count = \
            read_embeddings(opt.both_embeddings, skip_lines,
                            set_of_src_and_tgt_vocab)
        tgt_vectors = src_vectors
        logger.info("\tFound {} total vectors in file".format(total_vec_count))
    else:
        if opt.src_embeddings is not None:
            logger.info("Reading encoder embeddings from {}".format(
                opt.src_embeddings))
            src_vectors, total_vec_count = read_embeddings(
                opt.src_embeddings, skip_lines,
                filter_set=enc_vocab.stoi
            )
            logger.info("\tFound {} total vectors in file.".format(
                total_vec_count))
        else:
            src_vectors = None
        if opt.tgt_embeddings is not None:
            logger.info("Reading decoder embeddings from {}".format(
                opt.tgt_embeddings))
            tgt_vectors, total_vec_count = read_embeddings(
                opt.tgt_embeddings, skip_lines,
                filter_set=dec_vocab.stoi
            )
            logger.info(
                "\tFound {} total vectors in file".format(total_vec_count))
        else:
            tgt_vectors = None
    logger.info("After filtering to vectors in vocab:")
    if opt.src_embeddings is not None or opt.both_embeddings is not None:
        logger.info("\t* enc: %d match, %d missing, (%.2f%%)"
                    % calc_vocab_load_stats(enc_vocab, src_vectors))
    if opt.tgt_embeddings is not None or opt.both_embeddings is not None:
        logger.info("\t* dec: %d match, %d missing, (%.2f%%)"
                    % calc_vocab_load_stats(dec_vocab, tgt_vectors))

    # Write to file
    enc_output_file = opt.save_data + ".enc_embeddings.pt"
    dec_output_file = opt.save_data + ".dec_embeddings.pt"
    if opt.src_embeddings is not None or opt.both_embeddings is not None:
        logger.info("\nSaving encoder embeddings as:\n\t* enc: %s"
                    % enc_output_file)
        torch.save(
            convert_to_torch_tensor(src_vectors, enc_vocab),
            enc_output_file
        )
        # set the opt in place
        opt.pre_word_vecs_enc = enc_output_file
    if opt.tgt_embeddings is not None or opt.both_embeddings is not None:
        logger.info("\nSaving decoder embeddings as:\n\t* dec: %s"
                    % dec_output_file)
        torch.save(
            convert_to_torch_tensor(tgt_vectors, dec_vocab),
            dec_output_file
        )
        # set the opt in place
        opt.pre_word_vecs_dec = dec_output_file
