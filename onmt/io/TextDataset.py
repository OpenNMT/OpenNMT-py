# -*- coding: utf-8 -*-

from collections import Counter
import torch
import torchtext

from onmt.io.IO import ONMTDatasetBase, _join_dicts, _peek,\
                       _construct_example_fromlist


class TextDataset(ONMTDatasetBase):
    """ Dataset for data_type=='text' """

    def sort_key(self, ex):
        "Sort using the size of source example."
        return -len(ex.src)

    def _process_corpus(self, fields, src_examples_iter, tgt_examples_iter,
                        num_src_feats=0, num_tgt_feats=0,
                        src_seq_length=0, tgt_seq_length=0,
                        dynamic_dict=True, use_filter_pred=True):
        """
        Build Example objects, Field objects, and filter_pred function
        from text corpus.

        Args:
            fields: a dictionary of Field objects. Keys are like 'src',
                    'tgt', 'src_map', and 'alignment'.
            src_examples_iter: preprocessed source example_dict iterator.
            tgt_examples_iter: preprocessed target example_dict iterator.
            num_src_feats: number of source side features.
            num_tgt_feats: number of target side features.
            src_seq_length: maximum source sequence length.
            tgt_seq_length: maximum target sequence length.
            dynamic_dict: create dynamic dictionaries?
            use_filter_pred: use a custom filter predicate to filter examples?

        Returns:
            constructed tuple of Examples objects, Field objects, filter_pred.
        """
        self.data_type = 'text'

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.
        if tgt_examples_iter is not None:
            examples_iter = (_join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        if dynamic_dict:
            examples_iter = self._dynamic_dict(examples_iter)

        # Peek at the first to see which fields are used.
        ex, examples_iter = _peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)
        out_examples = (_construct_example_fromlist(ex_values, out_fields)
                        for ex_values in example_values)

        def filter_pred(example):
            return 0 < len(example.src) <= src_seq_length \
               and 0 < len(example.tgt) <= tgt_seq_length

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        return out_examples, out_fields, filter_pred

    def _dynamic_dict(self, examples_iter):
        for example in examples_iter:
            src = example["src"]
            src_vocab = torchtext.vocab.Vocab(Counter(src))
            self.src_vocabs.append(src_vocab)
            # Mapping source tokens to indices in the dynamic dict.
            src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
            example["src_map"] = src_map

            if "tgt" in example:
                tgt = example["tgt"]
                mask = torch.LongTensor(
                        [0] + [src_vocab.stoi[w] for w in tgt] + [0])
                example["alignment"] = mask
            yield example
