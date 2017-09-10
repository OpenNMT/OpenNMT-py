# -*- coding: utf-8 -*-

import argparse
import codecs
import torch

import onmt
import onmt.IO
import opts

parser = argparse.ArgumentParser(description='preprocess.py')
opts.add_md_help_argument(parser)


# **Preprocess Options**
parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-data_type', default="text",
                    help="Type of the source input. Options are [text|img].")
parser.add_argument('-data_img_dir', default=".",
                    help="Location of source images")

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                    help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")
parser.add_argument('-features_vocabs_prefix', type=str, default='',
                    help="Path prefix to existing features vocabularies")
parser.add_argument('-seed', type=int, default=3435,
                    help="Random seed")
parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opts.preprocess_opts(parser)

opt = parser.parse_args()
torch.manual_seed(opt.seed)


def main():
    print('Preparing training ...')
    with codecs.open(opt.train_src, "r", "utf-8") as src_file:
        src_line = src_file.readline().strip().split()
        _, _, nFeatures = onmt.IO.extract_features(src_line)

    fields = onmt.IO.ONMTDataset.get_fields(nFeatures)
    print("Building Training...")
    train = onmt.IO.ONMTDataset(opt.train_src, opt.train_tgt, fields, opt)
    print("Building Vocab...")
    onmt.IO.ONMTDataset.build_vocab(train, opt)

    print("Building Valid...")
    valid = onmt.IO.ONMTDataset(opt.valid_src, opt.valid_tgt, fields, opt)
    print("Saving train/valid/fields")

    # Can't save fields, so remove/reconstruct at training time.
    torch.save(onmt.IO.ONMTDataset.save_vocab(fields),
               open(opt.save_data + '.vocab.pt', 'wb'))
    train.fields = []
    valid.fields = []
    torch.save(train, open(opt.save_data + '.train.pt', 'wb'))
    torch.save(valid, open(opt.save_data + '.valid.pt', 'wb'))


if __name__ == "__main__":
    main()
