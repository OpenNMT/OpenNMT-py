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
    """ Checking if there are existing .pt files to avoid tampering """
    # We will use glob.glob() to find sharded {train|valid}.[0-9]*.pt
    # when training, so check to avoid tampering with existing pt files
    # or mixing them up.
    for t in ['train', 'valid', 'vocab']:
        pattern = opt.save_data + '.' + t + '*.pt'
        if glob.glob(pattern):
            sys.stderr.write("Please backup existing pt file: %s, "
                             "to avoid tampering!\n" % pattern)
            sys.exit(1)


def parse_args():
    """ Parsing arguments """
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
        if end is not None:
            f.writelines(data[start:end])
        else:
            f.writelines(data[start:])


def build_save_in_shards_using_shards_size(src_corpus, tgt_corpus, fields,
                                           corpus_type, opt):
    """
    Divide src_corpus and tgt_corpus into smaller multiples
    src_copus and tgt corpus files, then build shards, each
    shard will have opt.shard_size samples except last shard.

    The reason we do this is to avoid taking up too much memory due
    to sucking in a huge corpus file.
    """

    # Does this actually shard in a memory-efficient way? The two readlines()
    # calls at the beginning each read in the whole corpus at once. The result
    # should be efficient shards at training time, but we still need to read
    # in everything at once when we preprocess
    logger.info("Reading source and target files: %s %s."
                % (src_corpus, tgt_corpus))
    with codecs.open(src_corpus, "r", encoding="utf-8") as fsrc:
        src_data = fsrc.readlines()
    with codecs.open(tgt_corpus, "r", encoding="utf-8") as ftgt:
        tgt_data = ftgt.readlines()
    assert len(src_data) == len(tgt_data), \
        "Source and target should be the same length"

    num_shards = int(len(src_data) / opt.shard_size)
    for i in range(num_shards):
        logger.info("Splitting shard %d." % i)
        start = i * opt.shard_size
        end = start + opt.shard_size
        src_shard_path = src_corpus + ".{}.txt".format(i)
        _write_shard(src_shard_path, src_data, start, end)

        tgt_shard_path = tgt_corpus + ".{}.txt".format(i)
        _write_shard(tgt_shard_path, tgt_data, start, end)

    num_written = num_shards * opt.shard_size
    if len(src_data) > num_written:
        logger.info("Splitting shard %d." % num_shards)
        last_start = num_shards * opt.shard_size
        last_src_shard_path = src_corpus + ".{}.txt".format(num_shards)
        _write_shard(last_src_shard_path, src_data, last_start)

        last_tgt_shard_path = tgt_corpus + ".{}.txt".format(num_shards)
        _write_shard(last_tgt_shard_path, tgt_data, last_start)

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
    """ Building and saving the dataset """
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src
        tgt_corpus = opt.train_tgt
    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt

    if opt.shard_size > 0:
        return build_save_in_shards_using_shards_size(
            src_corpus, tgt_corpus, fields, corpus_type, opt)

    # For data_type == 'img' or 'audio', currently we don't do
    # preprocess sharding. We only build a monolithic dataset.
    # But since the interfaces are uniform, it would be not hard
    # to do this should users need this feature.
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
    """ Building and saving the vocab """
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency
    )

    # Can't save fields, so remove/reconstruct at training time.
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
