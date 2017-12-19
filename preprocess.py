#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import os

import torch

import onmt.io
import opts


def parse_args():
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    return opt


def get_num_features(side, opt):
    """ Peek one line and get number of features of it.
        (All lines must have same number of features).
    """
    assert side in ["src", "tgt"]

    # Only "text" corpus has srouce-side features.
    if side == "src":
        data_file = opt.train_src if opt.data_type == "text" else None
    else:
        # side == "tgt"
        data_file = opt.train_tgt

    if data_file is not None:
        with codecs.open(data_file, "r", "utf-8") as df:
            f_line = df.readline().strip().split()
            _, _, n_features = onmt.io.extract_features(f_line)
    else:
        n_features = 0

    return n_features


def build_save_text_dataset_in_shards(src_corpus, tgt_corpus,
                                      fields, corpus_type, opt):
    '''
    Divide the big text corpus into shards, and build dataset seperately.
    '''
    # The reason we do this is to avoid taking up too much memory due
    # to sucking in a huge corpus file.
    #
    # To tackle this, we only read in part of the corpus file of
    # size `max_shard_size`, and process it into dataset, then write
    # it to disk along the way. By doing this, we only focus on part
    # of the corpus at any moment, thus effectively reducing memory use.
    # According to test, this method can reduce memory footprint by ~40%.
    #
    # Note! As we process along the shards, previous shards might still
    # stay in memory, but since we are done with them, and no more
    # reference to them, if there is memory tight situation, the OS could
    # easily reclaim these memory.
    #
    # If `max_shard_size` is 0 or is larger than the corpus size, it is
    # effectively preprocessed into one dataset, i.e. no sharding.

    corpus_size = os.path.getsize(src_corpus)
    if corpus_size > 10 * (1024**2) and opt.max_shard_size == 0:
        print("Warning! The corpus %s is larger than 10M bytes, you can "
              "set '-max_shard_size' to process it by small shards "
              "to avoid memory hogging problem !!!" % src_corpus)

    ret_list = []
    src_iter = onmt.io.ShardedTextCorpusIterator(
                src_corpus, opt.src_seq_length_trunc,
                "src", opt.max_shard_size)
    tgt_iter = onmt.io.ShardedTextCorpusIterator(
                tgt_corpus, opt.tgt_seq_length_trunc,
                "tgt", opt.max_shard_size,
                assoc_iter=src_iter)

    index = 0
    while not src_iter.hit_end():
        index += 1
        dataset = onmt.io.TextDataset(
                fields, src_iter, tgt_iter,
                src_iter.n_feats, tgt_iter.n_feats,
                src_seq_length=opt.src_seq_length,
                tgt_seq_length=opt.tgt_seq_length,
                dynamic_dict=opt.dynamic_dict)
        ret_list.append(dataset)

        # We save fields in vocab.pt seperately, so make it empty.
        dataset.fields = []
        part_name = "{:s}.{:s}.{:d}.pt".format(
                    opt.save_data, corpus_type, index)
        torch.save(dataset, open(part_name, 'wb'))

    return ret_list


def build_save_dataset(corpus_type, fields, opt, save=True):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src
        tgt_corpus = opt.train_tgt
    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt

    if opt.data_type == 'text':
        # Currently we only do sharding for text corpus.
        return build_save_text_dataset_in_shards(
                src_corpus, tgt_corpus, fields, corpus_type, opt)

    # For data_type == 'img' or 'audio', currently we don't do
    # preprocess sharding. We only build a monolithic dataset.
    # But since the interfaces are uniform, it would be not hard
    # to do this should users need this feature.
    dataset = onmt.io.build_dataset(
                fields, opt.data_type, src_corpus, tgt_corpus,
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

    if save:
        # We save fields in vocab.pt seperately, so make it empty.
        dataset.fields = []
        pt_file = "{:s}.{:s}.pt".format(opt.save_data, corpus_type)
        torch.save(dataset, open(pt_file, 'wb'))

    return [dataset]


def build_save_vocab(train_dataset, fields, opt, save=True):
    # We've empty'ed each dataset's `fields` attribute
    # when saving datasets, so restore them.
    for train in train_dataset:
        train.fields = fields

    onmt.io.build_vocab(train_dataset, opt.data_type, opt.share_vocab,
                        opt.src_vocab_size,
                        opt.src_words_min_frequency,
                        opt.tgt_vocab_size,
                        opt.tgt_words_min_frequency)

    if save:
        # Can't save fields, so remove/reconstruct at training time.
        torch.save(onmt.io.save_fields_to_vocab(fields),
                   open(opt.save_data + '.vocab.pt', 'wb'))


def main():
    opt = parse_args()

    print('Preparing for training ...')
    n_src_features = get_num_features('src', opt)
    n_tgt_features = get_num_features('tgt', opt)
    fields = onmt.io.get_fields(opt.data_type, n_src_features, n_tgt_features)

    print("Building & saving training data...")
    train_datasets = build_save_dataset('train', fields, opt)

    print("Building & saving vocabulary...")
    build_save_vocab(train_datasets, fields, opt)

    print("Building & saving validation data...")
    build_save_dataset('valid', fields, opt)


if __name__ == "__main__":
    main()
