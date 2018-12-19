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
                             "to avoid overwriting them!\n" % path)
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


def _write_temp_shard_files(corpus, fields, corpus_type, shard_size):
    # Does this actually shard in a memory-efficient way? The readlines()
    # reads in the whole corpus. Shards should be efficient at training time,
    # but in principle it should not be necessary to read everything at once
    # when preprocessing either.
    with codecs.open(corpus, "r", encoding="utf-8") as f:
        data = f.readlines()
        corpus_size = len(data)

    if shard_size <= 0:
        shard_size = corpus_size
    for i, start in enumerate(range(0, corpus_size, shard_size)):
        logger.info("Splitting shard %d." % i)
        end = start + shard_size
        shard_path = corpus + ".{}.txt".format(i)
        _write_shard(shard_path, data, start, end)

    return corpus_size


def build_save_dataset(corpus_type, fields, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src = opt.train_src
        tgt = opt.train_tgt
    else:
        src = opt.valid_src
        tgt = opt.valid_tgt

    logger.info("Reading source and target files: %s %s." % (src, tgt))
    src_len = _write_temp_shard_files(src, fields, corpus_type, opt.shard_size)
    tgt_len = _write_temp_shard_files(tgt, fields, corpus_type, opt.shard_size)
    assert src_len == tgt_len, "Source and target should be the same length"

    src_shards = sorted(glob.glob(src + '.*.txt'))
    tgt_shards = sorted(glob.glob(tgt + '.*.txt'))
    shard_pairs = zip(src_shards, tgt_shards)
    dataset_paths = []

    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        logger.info("Building shard %d." % i)
        dataset = inputters.build_dataset(
            fields, opt.data_type,
            src=src_shard,
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            src_seq_len=opt.src_seq_length,
            tgt_seq_len=opt.tgt_seq_length,
            src_seq_length_trunc=opt.src_seq_length_trunc,
            tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
            dynamic_dict=opt.dynamic_dict,
            sample_rate=opt.sample_rate,
            window_size=opt.window_size,
            window_stride=opt.window_stride,
            window=opt.window,
            image_channel_size=opt.image_channel_size
        )

        data_path = "{:s}.{:s}.{:d}.pt".format(opt.save_data, corpus_type, i)
        dataset_paths.append(data_path)

        logger.info(" * saving %sth %s data shard to %s."
                    % (i, corpus_type, data_path))

        dataset.save(data_path)

        os.remove(src_shard)
        os.remove(tgt_shard)
        del dataset.examples
        gc.collect()
        del dataset
        gc.collect()

    return dataset_paths


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency
    )

    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(inputters.save_fields_to_vocab(fields), vocab_path)


def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


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

    src_nfeats = count_features(opt.train_src) if opt.data_type == 'text' \
        else 0
    tgt_nfeats = count_features(opt.train_tgt)  # tgt always text so far
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
