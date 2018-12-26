# -*- coding: utf-8 -*-

import torch

from onmt.inputters.dataset_base import DatasetBase


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

    @classmethod
    def make_examples(cls, sequences, truncate, side):
        """
        Args:
            sequences: path to corpus file or iterable
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        if isinstance(sequences, str):
            sequences = cls._read_file(sequences)
        for i, seq in enumerate(sequences):
            # the implicit assumption here is that data that does not come
            # from a file is already at least semi-tokenized, i.e. split on
            # whitespace. We cannot do modular/user-specified tokenization
            # until that is no longer the case. The fields should handle this.
            seq = seq.strip().split()
            if truncate:
                seq = seq[:truncate]

            words, feats, _ = TextDataset.extract_text_features(seq)

            example_dict = {side: words, "indices": i}
            if feats:
                prefix = side + "_feat_"
                example_dict.update((prefix + str(j), f)
                                    for j, f in enumerate(feats))
            yield example_dict
