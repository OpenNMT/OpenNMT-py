# -*- coding: utf-8 -*-

from itertools import chain
import torchtext
from onmt.Utils import aeq


PAD_WORD = '<blank>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'


class ONMTDatasetBase(torchtext.data.Dataset):
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

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(ONMTDatasetBase, self).__reduce_ex__()

    def collapse_copy_scores(self, scores, batch, tgt_vocab):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            index = batch.indices.data[b]
            src_vocab = self.src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    scores[:, b, ti] += scores[:, b, offset + i]
                    scores[:, b, offset + i].fill_(1e-20)
        return scores

    @staticmethod
    def coalesce_datasets(datasets):
        """Coalesce all dataset instances. """
        final = datasets[0]
        for d in datasets[1:]:
            # `src_vocabs` is a list of `torchtext.vocab.Vocab`.
            # Each sentence transforms into on Vocab.
            # Coalesce them into one big list.
            final.src_vocabs += d.src_vocabs

            # All datasets have same number of features.
            aeq(final.n_src_feats, d.n_src_feats)
            aeq(final.n_tgt_feats, d.n_tgt_feats)

            # `examples` is a list of `torchtext.data.Example`.
            # Coalesce them into one big list.
            final.examples += d.examples

            # All datasets have same fields, no need to update.

        return final

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

        split_tokens = [token.split(u"￨") for token in tokens]
        split_tokens = [token for token in split_tokens if token[0]]
        token_size = len(split_tokens[0])

        assert all(len(token) == token_size for token in split_tokens), \
            "all words must have the same number of features"
        words_and_features = list(zip(*split_tokens))
        words = words_and_features[0]
        features = words_and_features[1:]

        return words, features, token_size - 1

    # Below are helper functions for intra-class use only.

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
        ex = torchtext.data.Example()
        for (name, field), val in zip(fields, data):
            if field is not None:
                setattr(ex, name, field.preprocess(val))
            else:
                setattr(ex, name, val)
        return ex
