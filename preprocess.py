#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""

import configargparse
import glob
import sys
import gc
import os
import codecs
import torch
from onmt.utils.logging import init_logger, logger

import onmt.inputters as inputters
import onmt.opts as opts


def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid', 'vocab']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid ovewriting them!\n" % path)
            sys.exit(1)


def parse_args():
    parser = configargparse.ArgumentParser(
        description='preprocess.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    check_existing_pt_files(opt)

    return opt


def _write_shard(path, data, start, end=None):
    with codecs.open(path, "w", encoding="utf-8") as f:
        shard = data[start:end] if end is not None else data[start:]
        f.writelines(shard)


def _write_temp_shard_files(corpus, fields, corpus_type, opt):
    # Does this actually shard in a memory-efficient way? The readlines()
    # reads in the whole corpus. Shards should be efficient at training time,
    # but in principle it should not be necessary to read everything at once
    # when preprocessing either.
    with codecs.open(corpus, "r", encoding="utf-8") as f:
        data = f.readlines()
        corpus_size = len(data)

    num_shards = int(len(data) / opt.shard_size)
    for i in range(num_shards):
        logger.info("Splitting shard %d." % i)
        start = i * opt.shard_size
        end = start + opt.shard_size
        shard_path = corpus + ".{}.txt".format(i)
        _write_shard(shard_path, data, start, end)

    num_written = num_shards * opt.shard_size
    if len(data) > num_written:
        logger.info("Splitting shard %d." % num_shards)
        last_start = num_shards * opt.shard_size
        last_shard_path = corpus + ".{}.txt".format(num_shards)
        _write_shard(last_shard_path, data, last_start)
    return corpus_size


def build_save_in_shards(src_corpus, tgt_corpus, fields, corpus_type, opt):
    """
    Divide src_corpus and tgt_corpus into smaller portions of opt.shard_size
    samples (besides the last shard, which may be smaller).
    """

    logger.info("Reading source and target files: %s %s."
                % (src_corpus, tgt_corpus))

    src_len = _write_temp_shard_files(src_corpus, fields, corpus_type, opt)
    tgt_len = _write_temp_shard_files(tgt_corpus, fields, corpus_type, opt)
    assert src_len == tgt_len, "Source and target should be the same length"

    src_list = sorted(glob.glob(src_corpus + '.*.txt'))
    tgt_list = sorted(glob.glob(tgt_corpus + '.*.txt'))

    ret_list = []

    for i, src in enumerate(src_list):
        logger.info("Building shard %d." % i)
        dataset = inputters.build_dataset(
            fields, opt.data_type,
            src_path=src,
            tgt_path=tgt_list[i],
            src_dir=opt.src_dir,
            src_seq_length=opt.src_seq_length,
            tgt_seq_length=opt.tgt_seq_length,
            src_seq_length_trunc=opt.src_seq_length_trunc,
            tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
            dynamic_dict=opt.dynamic_dict,
            sample_rate=opt.sample_rate,
            window_size=opt.window_size,
            window_stride=opt.window_stride,
            window=opt.window,
            image_channel_size=opt.image_channel_size
        )

        pt_file = "{:s}.{:s}.{:d}.pt".format(opt.save_data, corpus_type, i)

        logger.info(" * saving %sth %s data shard to %s."
                    % (i, corpus_type, pt_file))
        dataset.save(pt_file)

        ret_list.append(pt_file)
        os.remove(src)
        os.remove(tgt_list[i])
        del dataset.examples
        gc.collect()
        del dataset
        gc.collect()

    return ret_list


def build_save_dataset(corpus_type, fields, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src
        tgt_corpus = opt.train_tgt
    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt

    if opt.shard_size > 0:
        return build_save_in_shards(
            src_corpus, tgt_corpus, fields, corpus_type, opt)

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
        window=opt.window,
        image_channel_size=opt.image_channel_size)

    pt_file = "{:s}.{:s}.pt".format(opt.save_data, corpus_type)
    logger.info(" * saving %s dataset to %s." % (corpus_type, pt_file))
    dataset.save(pt_file)

    return [pt_file]


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency
    )

    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(inputters.save_fields_to_vocab(fields), vocab_path)


def main():
    opt = parse_args()

    assert opt.max_shard_size == 0, \
        "-max_shard_size is deprecated. Please use \
        -shard_size (number of examples) instead."
    assert opt.shuffle == 0, \
        "-shuffle is not implemented. Please shuffle \
        your data before pre-processing."

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

    logger.info("Building & saving training data...")
    train_dataset_files = build_save_dataset('train', fields, opt)

    logger.info("Building & saving validation data...")
    build_save_dataset('valid', fields, opt)

    logger.info("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)


if __name__ == "__main__":
    main()
