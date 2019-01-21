"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from onmt.inputters.inputter import make_features, \
    load_old_vocab, get_fields, OrderedIterator, \
    build_dataset, build_vocab, old_style_vocab
from onmt.inputters.dataset import DatasetBase


__all__ = ['DatasetBase', 'make_features',
           'load_old_vocab', 'get_fields',
           'build_dataset', 'old_style_vocab',
           'build_vocab', 'OrderedIterator']
