"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from onmt.inputters.inputter import \
    load_old_vocab, get_fields, OrderedIterator, \
    build_dataset, build_vocab, old_style_vocab
from onmt.inputters.dataset_base import DatasetBase
from onmt.inputters.text_dataset import TextDataset, TextDataReader
from onmt.inputters.image_dataset import ImageDataset, ImageDataReader
from onmt.inputters.audio_dataset import AudioDataset, AudioDataReader
from onmt.inputters.datareader_base import DataReaderBase


str2reader = {
    "text": TextDataReader, "img": ImageDataReader, "audio": AudioDataReader}

__all__ = ['DatasetBase', 'load_old_vocab', 'get_fields', 'DataReaderBase',
           'build_dataset', 'old_style_vocab',
           'build_vocab', 'OrderedIterator',
           'TextDataset', 'ImageDataset', 'AudioDataset',
           'TextDataReader', 'ImageDataReader', 'AudioDataReader']
