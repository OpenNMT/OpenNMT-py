# -*- coding: utf-8 -*-
import os
import codecs
import torch
import pyonmttok
from onmt.constants import DefaultTokens


class IterOnDevice(object):
    """Sent items from `iterable` on `device_id` and yield."""

    def __init__(self, iterable, device_id):
        self.iterable = iterable
        self.device_id = device_id
        self.transforms = iterable.transforms

    @staticmethod
    def batch_to_device(tbatch, device_id):
        """Move `batch` to `device_id`, cpu if `device_id` < 0."""
        device = torch.device(device_id) if device_id >= 0 \
            else torch.device('cpu')
        for key in tbatch.keys():
            if key != 'src_ex_vocab':
                tbatch[key] = tbatch[key].to(device)

    def __iter__(self):
        for tbatch in self.iterable:
            self.batch_to_device(tbatch, self.device_id)
            yield tbatch


def build_vocab(opt):
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
    vocabs = {}
    src_vocab = _read_vocab_file(opt.src_vocab, opt.src_words_min_frequency)

    src_vocab = pyonmttok.build_vocab_from_tokens(
        src_vocab,
        maximum_size=opt.src_vocab_size,
        special_tokens=[DefaultTokens.UNK,
                        DefaultTokens.PAD,
                        DefaultTokens.BOS,
                        DefaultTokens.EOS])
    src_vocab.default_id = src_vocab[DefaultTokens.UNK]
    vocabs['src'] = src_vocab
    if opt.share_vocab:
        vocabs['tgt'] = src_vocab
    else:
        tgt_vocab = _read_vocab_file(opt.tgt_vocab,
                                     opt.tgt_words_min_frequency)
        tgt_vocab = pyonmttok.build_vocab_from_tokens(
            tgt_vocab,
            maximum_size=opt.tgt_vocab_size,
            special_tokens=[DefaultTokens.UNK,
                            DefaultTokens.PAD,
                            DefaultTokens.BOS,
                            DefaultTokens.EOS])
        tgt_vocab.default_id = tgt_vocab[DefaultTokens.UNK]
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
