# -*- coding: utf-8 -*-
import os
import math
import codecs
import torch
import pyonmttok
from onmt.constants import DefaultTokens


class IterOnDevice(torch.utils.data.IterableDataset):
    """Sent items from `iterable` on `device_id` and yield."""

    def __init__(self, iterable, device_id):
        super(IterOnDevice).__init__()
        self.iterable = iterable
        self.device_id = device_id

    @staticmethod
    def batch_to_device(tensor_batch, device_id):
        """Move `batch` to `device_id`, cpu if `device_id` < 0."""
        device = torch.device(device_id) if device_id >= 0 \
            else torch.device('cpu')
        for key in tensor_batch.keys():
            if key != 'src_ex_vocab':
                tensor_batch[key] = tensor_batch[key].to(device)

    def __iter__(self):
        for tensor_batch in self.iterable:
            self.batch_to_device(tensor_batch, self.device_id)
            yield tensor_batch


def build_vocab(opt, specials):
    """ Build vocabs dict to be stored in the checkpoint
        based on vocab files having each line [token, count]
    Args:
        opt: src_vocab, tgt_vocab, src_feats_vocab
    Return:
        vocabs: {'src': pyonmttok.Vocab, 'tgt': pyonmttok.Vocab,
                 'src_feats' : {'feat0': pyonmttok.Vocab,
                                'feat1': pyonmttok.Vocab, ...},
                 'data_task': seq2seq or lm
                }
    """
    def _pad_vocab_to_multiple(vocab, multiple):
        vocab_size = len(vocab)
        if vocab_size % multiple == 0:
            return vocab
        target_size = int(math.ceil(vocab_size / multiple)) * multiple
        for i in range(target_size - vocab_size):
            vocab.add_token(DefaultTokens.VOCAB_PAD + str(i))
        return vocab

    vocabs = {}
    src_vocab = _read_vocab_file(opt.src_vocab, opt.src_words_min_frequency)

    src_specials = list(specials['src'])
    src_vocab = pyonmttok.build_vocab_from_tokens(
        src_vocab,
        maximum_size=opt.src_vocab_size,
        special_tokens=[DefaultTokens.UNK,
                        DefaultTokens.PAD,
                        DefaultTokens.BOS,
                        DefaultTokens.EOS] + src_specials)
    src_vocab.default_id = src_vocab[DefaultTokens.UNK]
    if opt.vocab_size_multiple > 1:
        src_vocab = _pad_vocab_to_multiple(src_vocab, opt.vocab_size_multiple)
    vocabs['src'] = src_vocab
    if opt.share_vocab:
        vocabs['tgt'] = src_vocab
    else:
        tgt_vocab = _read_vocab_file(opt.tgt_vocab,
                                     opt.tgt_words_min_frequency)
        tgt_specials = list(specials['tgt'])
        tgt_vocab = pyonmttok.build_vocab_from_tokens(
            tgt_vocab,
            maximum_size=opt.tgt_vocab_size,
            special_tokens=[DefaultTokens.UNK,
                            DefaultTokens.PAD,
                            DefaultTokens.BOS,
                            DefaultTokens.EOS] + tgt_specials)
        tgt_vocab.default_id = tgt_vocab[DefaultTokens.UNK]
        if opt.vocab_size_multiple > 1:
            tgt_vocab = _pad_vocab_to_multiple(tgt_vocab,
                                               opt.vocab_size_multiple)
        vocabs['tgt'] = tgt_vocab

    if opt.src_feats_vocab:
        src_feats = {}
        for feat_name, filepath in opt.src_feats_vocab.items():
            src_f_vocab = _read_vocab_file(filepath, 1)
            src_f_vocab = pyonmttok.build_vocab_from_tokens(
                src_f_vocab,
                maximum_size=0,
                minimum_frequency=1,
                special_tokens=[DefaultTokens.UNK,
                                DefaultTokens.PAD,
                                DefaultTokens.BOS,
                                DefaultTokens.EOS])
            src_f_vocab.default_id = src_f_vocab[DefaultTokens.UNK]
            if opt.vocab_size_multiple > 1:
                src_f_vocab = _pad_vocab_to_multiple(src_f_vocab,
                                                     opt.vocab_size_multiple)
            src_feats[feat_name] = src_f_vocab
        vocabs['src_feats'] = src_feats

    vocabs['data_task'] = opt.data_task

    return vocabs


def _read_vocab_file(vocab_path, min_count):
    """Loads a vocabulary from the given path.

    Args:
        vocab_path (str): Path to utf-8 text file containing vocabulary.
            Each token should be on a line, may followed with a count number
            seperate by space if `with_count`. No extra whitespace is allowed.
        min_count (int): retains only tokens with min_count frequency.
    """

    if not os.path.exists(vocab_path):
        raise RuntimeError(
            "Vocabulary not found at {}".format(vocab_path))
    else:
        with codecs.open(vocab_path, 'r', 'utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            first_line = lines[0].split(None, 1)
            has_count = (len(first_line) == 2 and first_line[-1].isdigit())
            if has_count:
                vocab = []
                for line in lines:
                    if int(line.split(None, 1)[1]) >= min_count:
                        vocab.append(line.split(None, 1)[0])
            else:
                vocab = [line.strip().split()[0] for line in lines]
            return vocab


def vocabs_to_dict(vocabs):
    """
    Convert a dict of pyonmttok vocabs
    into a plain text dict to be saved in the checkpoint
    """
    vocabs_dict = {}
    vocabs_dict['src'] = vocabs['src'].ids_to_tokens
    vocabs_dict['tgt'] = vocabs['tgt'].ids_to_tokens
    if 'src_feats' in vocabs.keys():
        vocabs_dict['src_feats'] = {}
        for feat in vocabs['src_feats'].keys():
            vocabs_dict['src_feats'][feat] = \
                vocabs['src_feats'][feat].ids_to_tokens
    vocabs_dict['data_task'] = vocabs['data_task']
    return vocabs_dict


def dict_to_vocabs(vocabs_dict):
    """
    Convert a dict formatted vocabs (as stored in a checkpoint)
    into a dict of pyonmttok vocabs objects.
    """
    vocabs = {}
    vocabs['data_task'] = vocabs_dict['data_task']
    vocabs['src'] = pyonmttok.build_vocab_from_tokens(vocabs_dict['src'])
    if vocabs_dict['src'] == vocabs_dict['tgt']:
        vocabs['tgt'] = vocabs['src']
    else:
        vocabs['tgt'] = pyonmttok.build_vocab_from_tokens(vocabs_dict['tgt'])
    if 'src_feats' in vocabs_dict.keys():
        vocabs['src_feats'] = {}
        for feat in vocabs_dict['src_feats'].keys():
            vocabs['src_feats'][feat] = \
                pyonmttok.build_vocab_from_tokens(
                    vocabs_dict['src_feats'][feat])
    return vocabs
