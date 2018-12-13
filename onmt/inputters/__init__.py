"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from onmt.inputters.inputter import make_features, collect_features, \
    load_fields_from_vocab, get_fields, OrderedIterator, \
    save_fields_to_vocab, build_dataset, build_vocab
from onmt.inputters.dataset_base import DatasetBase, PAD_WORD, BOS_WORD, \
    EOS_WORD
from onmt.inputters.text_dataset import TextDataset
from onmt.inputters.image_dataset import ImageDataset
from onmt.inputters.audio_dataset import AudioDataset


__all__ = ['PAD_WORD', 'BOS_WORD', 'EOS_WORD', 'DatasetBase',
           'make_features', 'collect_features',
           'load_fields_from_vocab', 'get_fields',
           'save_fields_to_vocab', 'build_dataset',
           'build_vocab', 'OrderedIterator',
           'TextDataset', 'ImageDataset', 'AudioDataset']
