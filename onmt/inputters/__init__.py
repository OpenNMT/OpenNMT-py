"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from onmt.inputters.inputter import make_features, \
    load_old_vocab, get_fields, OrderedIterator, \
    build_dataset, build_vocab, old_style_vocab
from onmt.inputters.dataset import DatasetBase
from onmt.datatypes.text_datatype import TextDataset
from onmt.datatypes.image_datatype import ImageDataset
from onmt.datatypes.audio_datatype import AudioDataset


__all__ = ['DatasetBase', 'make_features',
           'load_old_vocab', 'get_fields',
           'build_dataset', 'old_style_vocab',
           'build_vocab', 'OrderedIterator',
           'TextDataset', 'ImageDataset', 'AudioDataset']
