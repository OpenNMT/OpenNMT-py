# -*- coding: utf-8 -*-

import onmt
import onmt.Markdown
import onmt.IO
import argparse
import torch
import dill

from train_opts import add_preprocess_arguments

parser = argparse.ArgumentParser(description='preprocess.py')
onmt.Markdown.add_md_help_argument(parser)

# **Preprocess Options**
parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-src_type', default="text",
                    help="Type of the source input. Options are [text|img].")
parser.add_argument('-src_img_dir', default=".",
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

add_preprocess_arguments(parser)

opt = parser.parse_args()
torch.manual_seed(opt.seed)


def main():
    print('Preparing training ...')

    fields = onmt.IO.ONMTDataset.get_fields(opt.train_src, opt.train_tgt)
    print("Building Training...")
    train = onmt.IO.ONMTDataset(opt.train_src, opt.train_tgt, fields, opt)
    print("Building Vocab...")
    onmt.IO.ONMTDataset.build_vocab(train, opt)

    print("Building Valid...")
    valid = onmt.IO.ONMTDataset(opt.valid_src, opt.valid_tgt, fields, opt)

    print("Saving train/valid/fields")
    torch.save(train, open(opt.save_data + '.train.pt', 'wb'),
               pickle_module=dill)
    torch.save(valid, open(opt.save_data + '.valid.pt', 'wb'),
               pickle_module=dill)
    torch.save(fields, open(opt.save_data + '.fields.pt', 'wb'),
               pickle_module=dill)


if __name__ == "__main__":
    main()
