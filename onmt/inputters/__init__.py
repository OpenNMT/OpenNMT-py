"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from onmt.inputters.inputter import collect_feature_vocabs, make_features, \
    num_features, \
    get_fields, vocab_to_fields, \
    fields_to_vocab, build_dataset, \
    build_vocabs, merge_vocabs, OrderedIterator
from onmt.inputters.dataset_base import DatasetBase, PAD_WORD, BOS_WORD, \
    EOS_WORD, UNK
from onmt.inputters.text_dataset import TextDataset, ShardedTextCorpusIterator
from onmt.inputters.image_dataset import ImageDataset
from onmt.inputters.audio_dataset import AudioDataset


__all__ = ['PAD_WORD', 'BOS_WORD', 'EOS_WORD', 'UNK', 'DatasetBase',
           'collect_feature_vocabs', 'make_features',
           'num_features', 'get_fields', 'vocab_to_fields',
           'fields_to_vocab', 'build_dataset',
           'build_vocabs', 'merge_vocabs', 'OrderedIterator',
           'TextDataset', 'ImageDataset', 'AudioDataset',
           'ShardedTextCorpusIterator']
