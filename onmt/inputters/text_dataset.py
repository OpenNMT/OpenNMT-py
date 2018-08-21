# -*- coding: utf-8 -*-
"""Define word-based embedders."""

from collections import Counter
import io
import codecs
import sys

import torch
from torchtext.vocab import Vocab
from torchtext.data import Example

from onmt.inputters.dataset_base import DatasetBase, UNK_WORD, PAD_WORD
from onmt.utils.misc import aeq


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

    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 num_src_feats=0, num_tgt_feats=0,
                 src_seq_length=0, tgt_seq_length=0,
                 dynamic_dict=True, use_filter_pred=True):
        # self.src_vocabs: mutated in dynamic_dict, used in translation.py
        # at translation time, this is 1 shorter than it is in master, causing
        # an indexing error
        self.src_vocabs = []

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats
        # another question: why is the Dataset constructor used in preprocess
        # and translate but not train?

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.
        if tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        if dynamic_dict:
            examples_iter = (self._dynamic_dict(ex) for ex in examples_iter)

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        # why are the fields in the examples_iter different from the ones
        # in the fields argument?
        fields = [(k, fields[k]) if k in fields else (k, None) for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        examples = [Example.fromlist(ev, fields) for ev in example_values]
        # the examples list for validation is one shorter at test time than
        # at preprocessing time
        # test and validation data are handled differently, I suspect, and this
        # has consequences.
        # at preprocessing time, inputters.build_dataset is never called in the
        # text case.
        # the iter arguments passed also originate in different places and have
        # different types
        # print(len(self.src_vocabs))

        def filter_pred(ex):
            """ ? """
            return 0 < len(ex.src) <= src_seq_length \
                and 0 < len(ex.tgt) <= tgt_seq_length

        filter_pred = filter_pred if use_filter_pred else None

        super(TextDataset, self).__init__(examples, fields, filter_pred)

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
    def make_examples_nfeats_tpl(cls, iterator, path,
                                 truncate, side, **kwargs):
        """
        Args:
            text_iter(iterator): an iterator (or None) that we can loop over
                to read examples.
                It may be an openned file, a string list etc...
            text_path(str): path to file or None
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Returns:
            (example_dict iterator, num_feats) tuple.
        """
        assert side in ['src', 'tgt']

        if iterator is None and path is None:
            return None, 0
        if path is not None:
            iterator = cls.make_iterator_from_file(path)

        examples_nfeats_iter = cls.make_examples(iterator, truncate, side)
        (_, num_feats), examples_nfeats_iter = cls._peek(examples_nfeats_iter)
        examples_iter = (ex for ex, nfeats in examples_nfeats_iter)

        return examples_iter, num_feats

    @classmethod
    def make_examples(cls, text_iter, truncate, side):
        """
        Args:
            text_iter (iterator): iterator of text sequences
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Yields:
            dict, int pairs where the dict is example stuff and the int is
            the number of features
        """
        # doesn't make examples.
        # this and the analogous methods in the other datasets are
        # used only in the make_examples_nfeats_tpl methods
        for i, line in enumerate(text_iter):
            line = line.strip().split()
            if truncate:
                line = line[:truncate]

            words, feats, n_feats = extract_text_features(line)

            example_dict = {side: words, "indices": i}
            if feats:
                prefix = side + "_feat_"
                example_dict.update((prefix + str(j), f)
                                    for j, f in enumerate(feats))
            yield example_dict, n_feats

    @classmethod
    def make_iterator_from_file(cls, path):
        with codecs.open(path, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                yield line

    # Below are helper functions for intra-class use only.
    def _dynamic_dict(self, example):
        """
        examples_iter: (self._join_dicts(src, tgt) for src, tgt in
                        zip(src_examples_iter, tgt_examples_iter))
        yields: dicts.
        """
        # this basically says that if you're using dynamic dict, you're
        # going to need some extra fields, and they get created in a
        # different way from other fields.
        src = example["src"]
        src_vocab = Vocab(Counter(src), specials=[UNK_WORD, PAD_WORD])
        self.src_vocabs.append(src_vocab)
        # Mapping source tokens to indices in the dynamic dict.
        src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
        example["src_map"] = src_map

        if "tgt" in example:
            tgt = example["tgt"]
            mask = torch.LongTensor(
                [0] + [src_vocab.stoi[w] for w in tgt] + [0])
            example["alignment"] = mask
        return example


class ShardedTextCorpusIterator(object):
    """
    This is the iterator for text corpus, used for sharding large text
    corpus into small shards, to avoid hogging memory.

    Inside this iterator, it automatically divides the corpus file into
    shards of size `shard_size`. Then, for each shard, it processes
    into (example_dict, n_features) tuples when iterates.
    """

    def __init__(self, corpus_path, line_truncate, side, shard_size,
                 assoc_iter=None):
        """
        Args:
            corpus_path: the corpus file path.
            line_truncate: the maximum length of a line to read.
                            0 for unlimited.
            side: "src" or "tgt".
            shard_size: the shard size, 0 means not sharding the file.
            assoc_iter: if not None, it is the associate iterator that
                        this iterator should align its step with.
        """
        try:
            # The codecs module seems to have bugs with seek()/tell(),
            # so we use io.open().
            self.corpus = io.open(corpus_path, "r", encoding="utf-8")
        except IOError:
            sys.stderr.write("Failed to open corpus file: %s" % corpus_path)
            sys.exit(1)

        self.line_truncate = line_truncate
        self.side = side
        self.shard_size = shard_size
        self.assoc_iter = assoc_iter
        self.last_pos = 0
        self.line_index = -1
        self.eof = False

    def __iter__(self):
        """
        Iterator of (example_dict, nfeats).
        On each call, it iterates over as many (example_dict, nfeats) tuples
        until this shard's size equals to or approximates `self.shard_size`.
        """
        iteration_index = -1
        if self.assoc_iter is not None:
            # We have associate iterator, just yields tuples
            # util we run parallel with it.
            while self.line_index < self.assoc_iter.line_index:
                line = self.corpus.readline()
                if line == '':
                    raise AssertionError(
                        "The two corpora must have same number of lines!")

                self.line_index += 1
                iteration_index += 1
                yield self._example_dict_iter(line, iteration_index)

            if self.assoc_iter.eof:
                self.eof = True
                self.corpus.close()
        else:
            # Yield tuples util this shard's size reaches the threshold.
            self.corpus.seek(self.last_pos)
            while True:
                if self.shard_size != 0 and self.line_index % 64 == 0:
                    # This part of check is time consuming on Py2 (but
                    # it is quite fast on Py3, weird!). So we don't bother
                    # to check for very line. Instead we chekc every 64
                    # lines. Thus we are not dividing exactly per
                    # `shard_size`, but it is not too much difference.
                    cur_pos = self.corpus.tell()
                    if cur_pos >= self.last_pos + self.shard_size:
                        self.last_pos = cur_pos
                        raise StopIteration

                line = self.corpus.readline()
                if line == '':
                    self.eof = True
                    self.corpus.close()
                    raise StopIteration

                self.line_index += 1
                iteration_index += 1
                yield self._example_dict_iter(line, iteration_index)

    def hit_end(self):
        """ ? """
        return self.eof

    @property
    def num_feats(self):
        """
        We peek the first line and seek back to
        the beginning of the file.
        """
        saved_pos = self.corpus.tell()

        line = self.corpus.readline().split()
        if self.line_truncate:
            line = line[:self.line_truncate]
        _, _, self.n_feats = extract_text_features(line)

        self.corpus.seek(saved_pos)

        return self.n_feats

    def _example_dict_iter(self, line, index):
        line = line.split()
        if self.line_truncate:
            line = line[:self.line_truncate]
        words, feats, n_feats = extract_text_features(line)
        example_dict = {self.side: words, "indices": index}
        if feats:
            # All examples must have same number of features.
            aeq(self.n_feats, n_feats)

            prefix = self.side + "_feat_"
            example_dict.update((prefix + str(j), f)
                                for j, f in enumerate(feats))

        return example_dict
