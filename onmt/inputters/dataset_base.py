# coding: utf-8

from itertools import chain
from collections import Counter
import codecs

import torch
import torchtext
from torchtext.vocab import Vocab

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


class DatasetBase(torchtext.data.Dataset):
    """
    A dataset basically supports iteration over all the examples
    it contains. We currently have 3 datasets inheriting this base
    for 3 types of corpus respectively: "text", "img", "audio".

    Internally it initializes an `torchtext.data.Dataset` object with
    the following attributes:

     `examples`: a sequence of `torchtext.data.Example` objects.
     `fields`: a dictionary associating str keys with `torchtext.data.Field`
        objects, and not necessarily having the same keys as the input fields.
    """

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, _d):
        self.__dict__.update(_d)

    def __reduce_ex__(self, proto):
        # This is a hack. Something is broken with torch pickle.
        return super(DatasetBase, self).__reduce_ex__()

    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 dynamic_dict=False, filter_pred=None):

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.
        if tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        # self.src_vocabs is used in collapse_copy_scores and in Translator.py
        self.src_vocabs = []
        if dynamic_dict:
            unk, pad = fields['src'].unk_token, fields['src'].pad_token
            examples_iter = (self._dynamic_dict(ex, unk, pad)
                             for ex in examples_iter)

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        # why do we need to use different keys from the ones passed in?
        fields = [(k, fields[k]) if k in fields else (k, None) for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)
        examples = [self._construct_example_fromlist(ex_values, fields)
                    for ex_values in example_values]

        super(DatasetBase, self).__init__(examples, fields, filter_pred)

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)

    @staticmethod
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

        specials = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]
        words = []
        features = []
        n_feats = None
        for token in tokens:
            split_token = token.split(u"￨")
            assert all([special != split_token[0] for special in specials]), \
                "Dataset cannot contain Special Tokens"

            if split_token[0]:
                words += [split_token[0]]
                features += [split_token[1:]]

                if n_feats is None:
                    n_feats = len(split_token)
                assert len(split_token) == n_feats, \
                    "all words must have the same number of features"
        features = list(zip(*features))
        return tuple(words), features, n_feats - 1

    def _join_dicts(self, *args):
        """
        Args:
            dictionaries with disjoint keys.

        Returns:
            a single dictionary that has the union of these keys.
        """
        return dict(chain(*[d.items() for d in args]))

    def _peek(self, seq):
        """
        Args:
            seq: an iterator.

        Returns:
            the first thing returned by calling next() on the iterator
            and an iterator created by re-chaining that value to the beginning
            of the iterator.
        """
        first = next(seq)
        return first, chain([first], seq)

    def _construct_example_fromlist(self, data, fields):
        """
        Args:
            data: the data to be set as the value of the attributes of
                the to-be-created `Example`, associating with respective
                `Field` objects with same key.
            fields: a dict of `torchtext.data.Field` objects. The keys
                are attributes of the to-be-created `Example`.

        Returns:
            the created `Example` object.
        """
        # why does this exist?
        ex = torchtext.data.Example()
        for (name, field), val in zip(fields, data):
            if field is not None:
                setattr(ex, name, field.preprocess(val))
            else:
                setattr(ex, name, val)
        return ex

    def _dynamic_dict(self, example, unk, pad):
        # it would not be necessary to pass unk and pad if the method were
        # called after fields becomes an attribute of self
        src = example["src"]
        src_vocab = Vocab(Counter(src), specials=[unk, pad])
        self.src_vocabs.append(src_vocab)
        # Map source tokens to indices in the dynamic dict.
        src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
        example["src_map"] = src_map

        if "tgt" in example:
            tgt = example["tgt"]
            mask = torch.LongTensor(
                [0] + [src_vocab.stoi[w] for w in tgt] + [0])
            example["alignment"] = mask
        return example

    @classmethod
    def _read_file(cls, path):
        with codecs.open(path, "r", "utf-8") as f:
            for line in f:
                yield line
