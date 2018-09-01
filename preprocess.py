#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pre-process Data / features files and build vocabulary
"""

import argparse
import os
import glob
import sys

import torch

from onmt.utils.logging import init_logger, logger

import onmt.inputters as inputters
import onmt.opts as opts


def check_existing_pt_files(opt):
    """ Checking if there are existing .pt files to avoid tampering """
    for t in ['train', 'valid', 'vocab']:
        pattern = opt.save_data + '.' + t + '*.pt'
        if glob.glob(pattern):
            sys.stderr.write("Please back up existing pt file: %s, "
                             "to avoid tampering!\n" % pattern)
            sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    check_existing_pt_files(opt)

    return opt


def build_datasets(corpus_type, fields, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src
        tgt_corpus = opt.train_tgt
    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt

    shard_size = opt.max_shard_size if corpus_type == 'train' else -1
    corpus_size = os.path.getsize(src_corpus)
    if corpus_type == 'train' and corpus_size > 10 * 1024 ** 2 \
            and shard_size == -1:
        logger.info("Warning. The corpus %s is larger than 10M bytes, "
                    "you can set '-max_shard_size' to process it in "
                    "small shards to use less memory." % src_corpus)

    if shard_size > 0:
        logger.info(' * divide corpus into shards and build dataset '
                    'separately (shard_size = %d bytes).' % shard_size)
    shards = inputters.shard_corpus(src_corpus, tgt_corpus, shard_size)

    ret_list = []
    for src_shard, tgt_shard in shards:
        dataset = inputters.build_dataset(
            fields, opt.data_type,
            src_lines=src_shard, tgt_lines=tgt_shard,
            src_dir=opt.src_dir,
            src_seq_length=opt.src_seq_length,
            tgt_seq_length=opt.tgt_seq_length,
            src_seq_length_trunc=opt.src_seq_length_trunc,
            tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
            dynamic_dict=opt.dynamic_dict,
            sample_rate=opt.sample_rate,
            window_size=opt.window_size,
            window_stride=opt.window_stride,
            window=opt.window)
        ret_list.append(dataset)
    return ret_list


def save_datasets(datasets, corpus_type, save_data):
    for i, dataset in enumerate(datasets, 1):
        if len(datasets) > 1:
            pt_file = "{:s}.{:s}.{:d}.pt".format(save_data, corpus_type, i)
        else:
            pt_file = "{:s}.{:s}.pt".format(save_data, corpus_type)
        logger.info(" * saving %s dataset to %s." % (corpus_type, pt_file))
        torch.save(dataset, pt_file)


def build_vocab(train_dataset, opt):
    fields = inputters.build_vocabs(
        train_dataset, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency)
    return inputters.fields_to_vocab(fields)


def main():
    opt = parse_args()
    init_logger(opt.log_file)

    logger.info("Extracting features...")
    src_nfeats = inputters.num_features(opt.train_src) \
        if opt.data_type == 'text' else 0
    tgt_nfeats = inputters.num_features(opt.train_tgt)
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(
        opt.data_type, src_nfeats, tgt_nfeats, opt.dynamic_dict)

    logger.info("Building training data and vocabulary...")

    train_datasets = build_datasets('train', fields, opt)
    vocab = build_vocab(train_datasets, opt)

    logger.info("Saving training data and vocabulary...")
    save_datasets(train_datasets, 'train', opt.save_data)
    torch.save(vocab, opt.save_data + '.vocab.pt')

    logger.info("Building & saving validation data...")
    valid_datasets = build_datasets('valid', fields, opt)
    save_datasets(valid_datasets, 'valid', opt.save_data)


if __name__ == "__main__":
    main()
