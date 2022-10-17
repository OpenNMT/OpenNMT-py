"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of vectors.
"""
from onmt.inputters.inputter import build_vocab, IterOnDevice
from onmt.inputters.text_utils import text_sort_key, max_tok_len, process,\
    numericalize, tensorify


__all__ = ['IterOnDevice', 'build_vocab', 'text_sort_key', 'max_tok_len',
           'process', 'numericalize', 'tensorify']
