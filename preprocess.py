#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import torch

import onmt
import onmt.IO
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
            _, _, n_features = onmt.IO.extract_features(f_line)
    else:
        n_features = 0

    return n_features


def build_dataset(corpus_type, fields, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src
        tgt_corpus = opt.train_tgt
    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt

    dataset = onmt.IO.build_dataset(
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

    return dataset


def main():
    opt = parse_args()

    print('Preparing for training ...')
    n_src_features = get_num_features('src', opt)
    n_tgt_features = get_num_features('tgt', opt)
    fields = onmt.IO.get_fields(opt.data_type, n_src_features, n_tgt_features)

    print("Building training data...")
    train = build_dataset('train', fields, opt)

    print("Building vocabulary...")
    onmt.IO.build_vocab(train, opt.data_type, opt.share_vocab,
                        opt.src_vocab_size,
                        opt.src_words_min_frequency,
                        opt.tgt_vocab_size,
                        opt.tgt_words_min_frequency)

    print("Building validation data...")
    valid = build_dataset('valid', fields, opt)

    print("Saving train/valid/vocab...")
    # Can't save fields, so remove/reconstruct at training time.
    torch.save(onmt.IO.save_vocab(fields),
               open(opt.save_data + '.vocab.pt', 'wb'))
    train.fields = []
    valid.fields = []
    torch.save(train, open(opt.save_data + '.train.pt', 'wb'))
    torch.save(valid, open(opt.save_data + '.valid.pt', 'wb'))


if __name__ == "__main__":
    main()
