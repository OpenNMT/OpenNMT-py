# -*- coding: utf-8 -*-

import io

import torch

from onmt.inputters.dataset_base import DatasetBase


def extract_text_features(tokens):
    """
    Args:
        tokens: A list of tokens, where each token consists of a word,
            optionally followed by u"￨"-delimited features.
    Returns:
        A sequence of words, a sequence of features, and num of features.
    """
    if not tokens:
        return [], [], -1

    split_tokens = [token.split(u"￨") for token in tokens]
    split_tokens = [token for token in split_tokens if token[0]]
    token_size = len(split_tokens[0])

    assert all(len(token) == token_size for token in split_tokens), \
        "all words must have the same number of features"
    words_and_features = list(zip(*split_tokens))
    words = words_and_features[0]
    features = words_and_features[1:]

    return words, features, token_size - 1


class TextDataset(DatasetBase):
    """ Dataset for data_type=='text'

        Build `Example` objects, `Field` objects, and filter_pred function
        from text corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
                Keys are like 'src', 'tgt', 'src_map', and 'alignment'.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            src_seq_length (int): maximum source sequence length.
            tgt_seq_length (int): maximum target sequence length.
            dynamic_dict (bool): create dynamic dictionaries?
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """
    data_type = 'text'

    @staticmethod
    def sort_key(ex):
        """ Sort using length of source sentences. """
        # Default to a balanced sort, prioritizing tgt len match.
        # TODO: make this configurable.
        if hasattr(ex, "tgt"):
            return len(ex.src), len(ex.tgt)
        return len(ex.src)

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        # this is a static method, but at least one of the arguments is always
        # an attribute of a TextDataset instance. It is not clear why this
        # is in the Dataset at all because it's used for computing scores,
        # not anything relating to iterating over the data.
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            blank = []
            fill = []
            index = batch.indices[b]
            src_vocab = src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices)
                fill = torch.Tensor(fill).type_as(batch.indices)
                scores[:, b].index_add_(1, fill,
                                        scores[:, b].index_select(1, blank))
                scores[:, b].index_fill_(1, blank, 1e-10)
        return scores

    @classmethod
    def _make_example(cls, line, truncate, side, **kwargs):
        line = line.strip().split()
        if truncate:
            line = line[:truncate]

        words, feats, _ = extract_text_features(line)

        example_dict = {side: words}
        if feats:
            prefix = side + "_feat_"
            example_dict.update((prefix + str(j), f)
                                for j, f in enumerate(feats))
        return example_dict

    @classmethod
    def _make_iterator_from_file(cls, path, **kwargs):
        with io.open(path, "r", encoding="utf-8") as corpus_file:
            for line in corpus_file:
                yield line
