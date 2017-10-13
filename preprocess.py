# -*- coding: utf-8 -*-

import configargparse
import codecs
import torch

import onmt
import onmt.IO
import opts

parser = configargparse.ArgumentParser(description='preprocess.py')
opts.add_md_help_argument(parser)
opts.config_opts(parser)
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
