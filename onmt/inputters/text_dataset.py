# -*- coding: utf-8 -*-

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

    @staticmethod
    def sort_key(ex):
        if hasattr(ex, "tgt"):
            return len(ex.src), len(ex.tgt)
        return len(ex.src)

    @classmethod
    def make_examples(cls, sequences, side):
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
            yield {side: seq, "indices": i}
