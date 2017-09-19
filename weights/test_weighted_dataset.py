import argparse
import sys
import random
import os
import torch
import codecs

import onmt.IO
import opts
from weights.weighted_dataset import ONMTWeightedDataset


def preprocess_args(parser):
    """
    replicate the arg parser on the module scope of IO
    """
    opts.add_md_help_argument(parser)

    # **Preprocess Options**
    parser.add_argument('-config', help="Read options from this file")

    parser.add_argument('-data_type', default="text",
                        help="Type of the source input." +
                             "Options are [text|img].")
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


def create_dw_datasets():
    """
    create the dummy datum-weight dataset for testing the functions
    """
    source_train_file = "../data/src-train.txt"
    source_val_file = "../data/src-val.txt"
    weight_train_file = "../data/dw-train.txt"
    weight_val_file = "../data/dw-val.txt"

    with open(weight_train_file, 'w') as destination:
        with open(source_train_file, 'r') as origin:
            for line in origin:
                destination.write(str(random.random()) + "\n")

    with open(weight_val_file, 'w') as destination:
        with open(source_val_file, 'r') as origin:
            for line in origin:
                destination.write(str(random.random()) + "\n")
    return weight_train_file, weight_val_file


def test_create_datasets(weight_train_file, weight_val_file):
    sys.argv.extend(["-train_src", "../data/src-train.txt",
                     "-train_tgt", "../data/tgt-train.txt",
                     "-valid_src", "../data/src-val.txt",
                     "-valid_tgt", "../data/tgt-val.txt",
                     "-save_data", "data_tests",
                     "-train_dw", weight_train_file,
                     "-valid_dw", weight_val_file])
    parser = argparse.ArgumentParser(description='preprocess.py')
    preprocess_args(parser)
    opts.model_opts(parser)

    parser.add_argument('-train_dw', default=None,
                        help="Path to the training datum-weight data")
    parser.add_argument('-valid_dw', default=None,
                        help="Path to the validation datum-weight data")

    opt = parser.parse_args()
    # torch.manual_seed(opt.seed)

    print('Preparing training ...')
    with codecs.open(opt.train_src, "r", "utf-8") as src_file:
        src_line = src_file.readline().strip().split()
        _, _, nFeatures = onmt.IO.extract_features(src_line)

    fields = ONMTWeightedDataset.get_fields(nFeatures)
    print("Building Training...")
    train = ONMTWeightedDataset(opt.train_src, opt.train_tgt,
                                fields, opt, dw_path=opt.train_dw)
    print("Building Vocab...")
    ONMTWeightedDataset.build_vocab(train, opt)
    print(train)

    print("Building Valid...")
    valid = ONMTWeightedDataset(opt.valid_src, opt.valid_tgt,
                                fields, opt, dw_path=opt.valid_dw)
    print("Saving train/valid/fields")

    # Can't save fields, so remove/reconstruct at training time.
    torch.save(ONMTWeightedDataset.save_vocab(fields),
               open(opt.save_data + '.vocab.pt', 'wb'))
    train.fields = []
    valid.fields = []
    torch.save(train, open(opt.save_data + '.train.pt', 'wb'))
    torch.save(valid, open(opt.save_data + '.valid.pt', 'wb'))
    return opt.save_data


def test_load_datasets(train_path):
    """
    replicate the loading aspect of train.py to verify reconstructed dataset
    """
    sys.argv.extend(["-data", train_path])

    parser = argparse.ArgumentParser(description='train.py')

    # Data and loading options
    parser.add_argument('-data', required=True,
                        help='Path to the *-train.pt file from preprocess.py')

    # opts.py
    opts.add_md_help_argument(parser)
    opts.train_opts(parser)
    opts.model_opts(parser)
    opt = parser.parse_args()

    print("Loading data from '%s'" % opt.data)

    train = torch.load(opt.data + '.train.pt')
    fields = ONMTWeightedDataset.load_fields(
        torch.load(opt.data + '.vocab.pt'))
    valid = torch.load(opt.data + '.valid.pt')
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in train.examples[0].__dict__])
    train.fields = fields
    valid.fields = fields
    src_features = [fields["src_feat_"+str(j)]
                    for j in range(train.nfeatures)]
    model_opt = opt
    checkpoint = None

    print(' * vocabulary size. source = %d; target = %d' %
          (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' %
              (j, len(feat.vocab)))

    print(' * number of training sentences. %d' %
          len(train))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')
    model = onmt.Models.make_base_model(opt, model_opt, fields, checkpoint)
    print(model)

    if not opt.train_from:
        if opt.param_init != 0.0:
            print('Intializing params')
            for p in model.parameters():
                p.data.uniform_(-opt.param_init, opt.param_init)
        model.encoder.embeddings.load_pretrained_vectors(opt.pre_word_vecs_enc,
                                                         opt.fix_word_vecs_enc)
        model.decoder.embeddings.load_pretrained_vectors(opt.pre_word_vecs_dec,
                                                         opt.fix_word_vecs_dec)

        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            opt=opt
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)

    if opt.train_from:
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    optim.set_parameters(model.parameters())

    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
        else:
            print(name, param.nelement())
    print('encoder: ', enc)
    print('decoder: ', dec)

    check_model_path(opt)

    # train_model(model, train, valid, fields, optim)


def check_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


if __name__ == "__main__":
    # Due to bypassing argsparse, its either the first two lines or
    # the last two. All of them together would fail.

    # Test ONMT dataset creation
    # weight_train_file, weight_val_file = create_dw_datasets()
    # train_path = test_create_datasets(weight_train_file, weight_val_file)

    # Test loading from the file for training
    train_path = "data_tests"
    test_load_datasets(train_path)
