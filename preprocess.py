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


def build_sharded_datasets(src_corpus, tgt_corpus, fields, corpus_type, opt):
    """
    Supported only if data_type == 'text'
    A large corpus is represented as a sequence of `shard` datasets: the
    corpus is read in small pieces of a size that is a multiple of 64 bytes
    >= `max_shard_size`. This can reduce the memory footprint by ~50%.

    Note! Previous shards may still remain in memory until they
    stay in memory, but since we are done with them, and no more
    reference to them, if there is memory tight situation, the OS could
    easily reclaim these memory.
    If `max_shard_size` is 0 or larger than the corpus size, no sharding is
    performed.
    NOTE! Because a dataset contains both source and target, a sharded output
    file is of size 2 * `max_shard_size` bytes.
    """
    corpus_size = os.path.getsize(src_corpus)
    if corpus_size > 10 * 1024 ** 2 and opt.max_shard_size == 0:
        logger.info("Warning. The corpus %s is larger than 10M bytes, "
                    "you can set '-max_shard_size' to process it in "
                    "small shards to use less memory." % src_corpus)

    if opt.max_shard_size != 0:
        logger.info(' * divide corpus into shards and build dataset '
                    'separately (shard_size = %d bytes).' % opt.max_shard_size)

    src_iter = inputters.ShardedTextCorpusIterator(
        src_corpus, opt.src_seq_length_trunc,
        "src", opt.max_shard_size)
    tgt_iter = inputters.ShardedTextCorpusIterator(
        tgt_corpus, opt.tgt_seq_length_trunc,
        "tgt", opt.max_shard_size,
        assoc_iter=src_iter)

    ret_list = []
    while not src_iter.hit_end():
        dataset = inputters.TextDataset(
            fields, src_iter, tgt_iter,
            src_iter.num_feats, tgt_iter.num_feats,
            src_seq_length=opt.src_seq_length,
            tgt_seq_length=opt.tgt_seq_length,
            dynamic_dict=opt.dynamic_dict)
        ret_list.append(dataset)

    return ret_list


def build_datasets(corpus_type, fields, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src
        tgt_corpus = opt.train_tgt
    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt

    # Currently preprocess sharding is only supported for data_type=='text'
    if opt.data_type == 'text':
        return build_sharded_datasets(
            src_corpus, tgt_corpus, fields,
            corpus_type, opt)

    # For data_type == 'img' or 'audio', preprocess sharding is not supported.
    # But since the interfaces are uniform, it should not be not hard to
    # implement this
    dataset = inputters.build_dataset(
        fields, opt.data_type,
        src_path=src_corpus,
        tgt_path=tgt_corpus,
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
    return [dataset]


def save_datasets(datasets, corpus_type, save_data):
    for i, dataset in enumerate(datasets, 1):
        dataset.fields = []  # fields probably CAN be saved, actually!

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
    src_nfeats = inputters.get_num_features(
        opt.data_type, opt.train_src, 'src')
    tgt_nfeats = inputters.get_num_features(
        opt.data_type, opt.train_tgt, 'tgt')
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(opt.data_type, src_nfeats, tgt_nfeats)

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
