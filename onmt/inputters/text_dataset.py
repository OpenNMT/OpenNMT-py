# -*- coding: utf-8 -*-

from collections import Counter
import codecs

import torch
from torchtext.vocab import Vocab

from onmt.inputters.dataset_base import DatasetBase, UNK_WORD, PAD_WORD


class TextDataset(DatasetBase):
    """
    Build `Example` objects, `Field` objects, and filter_pred function
    from text corpus.

    Args:
        fields (dict): a dictionary of `torchtext.data.Field`.
            Keys are like 'src', 'tgt', 'src_map', and 'alignment'.
        src_examples_iter (dict iter): preprocessed source example
            dictionary iterator.
        tgt_examples_iter (dict iter): preprocessed target example
            dictionary iterator.
        dynamic_dict (bool)
    """
    data_type = 'text'  # get rid of this class attribute asap

    @staticmethod
    def sort_key(ex):
        if hasattr(ex, "tgt"):
            return len(ex.src), len(ex.tgt)
        return len(ex.src)

    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 dynamic_dict=True, filter_pred=None):
        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.
        if tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        if dynamic_dict:
            examples_iter = self._dynamic_dict(examples_iter)

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        # why do we need to use different keys from the ones passed in?
        fields = [(k, fields[k]) if k in fields else (k, None) for k in keys]

        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        examples = [self._construct_example_fromlist(ex_values, fields)
                    for ex_values in example_values]

        super(TextDataset, self).__init__(examples, fields, filter_pred)

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs,
                             batch_dim=1, batch_offset=None):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambiguous.
        """
        offset = len(tgt_vocab)
        for b in range(scores.size(batch_dim)):
            blank = []
            fill = []
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            src_vocab = src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices.data)
                fill = torch.Tensor(fill).type_as(batch.indices.data)
                score = scores[:, b] if batch_dim == 1 else scores[b]
                score.index_add_(1, fill, score.index_select(1, blank))
                score.index_fill_(1, blank, 1e-10)
        return scores

    @staticmethod
    def make_text_examples(text_iter, text_path, truncate, side):
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
            example_dict iterator
        """
        # this will probably be removed soon
        assert side in ['src', 'tgt']

        if text_iter is None and text_path is None:
            return None

        if text_iter is None:
            text_iter = TextDataset.make_iterator_from_file(text_path)

        examples = TextDataset.make_examples(text_iter, truncate, side)

        return examples

    @staticmethod
    def make_examples(text_iter, truncate, side):
        """
        Args:
            text_iter (iterator): iterator of text sequences
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        # this function basically reimplements what torchtext Fields and
        # examples give you for free. It should not exist.
        for i, line in enumerate(text_iter):
            line = line.strip().split()
            if truncate:
                line = line[:truncate]

            words, feats, _ = TextDataset.extract_text_features(line)

            example_dict = {side: words, "indices": i}
            if feats:
                prefix = side + "_feat_"
                example_dict.update((prefix + str(j), f)
                                    for j, f in enumerate(feats))
            yield example_dict

    @staticmethod
    def make_iterator_from_file(path):
        with codecs.open(path, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                yield line

    def _dynamic_dict(self, examples_iter):
        for example in examples_iter:
            src = example["src"]
            src_vocab = Vocab(Counter(src), specials=[UNK_WORD, PAD_WORD])
            self.src_vocabs.append(src_vocab)
            # Map source tokens to indices in the dynamic dict.
            src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
            example["src_map"] = src_map

            if "tgt" in example:
                tgt = example["tgt"]
                mask = torch.LongTensor(
                    [0] + [src_vocab.stoi[w] for w in tgt] + [0])
                example["alignment"] = mask
            yield example
