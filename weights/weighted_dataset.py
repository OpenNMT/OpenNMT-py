import codecs
from collections import Counter, defaultdict
from itertools import chain, count
import argparse
import sys
import random
import os

import torch
import torchtext.data
import torchtext.vocab
from onmt.IO import PAD_WORD, UNK, BOS_WORD, EOS_WORD, __getstate__, __setstate__, extract_features, merge_vocabs, \
    make_features, join_dicts, OrderedIterator, ONMTDataset, load_image_libs
import onmt.IO
import opts

torchtext.vocab.Vocab.__getstate__ = __getstate__
torchtext.vocab.Vocab.__setstate__ = __setstate__


# TODO: inherit from ONMTDataset adding the target "weights"

class ONMTWeightedDataset(ONMTDataset):
    """Defines a dataset for machine translation with the choice of weighting
    the targets (or source-target pairs) for a measure of 'trustworthiness'."""

    def __init__(self, src_path, tgt_path, fields, opt,
                 src_img_dir=None, dw_path=None, **kwargs):
        """
        Create a TranslationDataset given paths and fields.

        src_path: location of source-side data
        tgt_path: location of target-side data or None. If it exists, it
                  source and target data must be the same length.
        fields:
        src_img_dir: if not None, uses images instead of text for the
                     source. TODO: finish
        """
        if src_img_dir:
            self.type_ = "img"
        else:
            self.type_ = "text"

        if self.type_ == "text":
            self.src_vocabs = []
            src_truncate = 0 if opt is None else opt.src_seq_length_trunc
            src_data = self._read_corpus_file(src_path, src_truncate)
            src_examples = self._construct_examples(src_data, "src")
            self.nfeatures = src_data[0][2]
        else:
            # TODO finish this.
            if not transforms:
                load_image_libs()

        if tgt_path:
            tgt_truncate = 0 if opt is None else opt.tgt_seq_length_trunc
            tgt_data = self._read_corpus_file(tgt_path, tgt_truncate)
            assert len(src_data) == len(tgt_data), \
                "Len src and tgt do not match"
            tgt_examples = self._construct_examples(tgt_data, "tgt")
        else:
            tgt_examples = None

        # datum-weights
        if tgt_path and dw_path:
            # dw_examples = [{"dw": 1} for _ in tgt_data]
            # dw_truncate = 0
            # dw_data = self._read_corpus_file(dw_path, dw_truncate)
            with codecs.open(dw_path, "r", "utf-8") as corpus_file:
                # lines = (line.split() for line in corpus_file)
                dw_data = [float(line) for line in corpus_file]

            assert len(src_data) == len(dw_data), \
                "Len src and dw do not match"
            dw_examples = [{"dw": dw} for dw in dw_data]
        else:
            dw_examples = None

        # examples: one for each src line or (src, tgt) line pair.
        # Each element is a dictionary whose keys represent at minimum
        # the src tokens and their indices and potentially also the
        # src and tgt features and alignment information.
        if tgt_examples and dw_path:
            examples = [join_dicts(src, tgt, dw)
                        for src, tgt, dw in zip(src_examples, tgt_examples, dw_examples)]
        elif tgt_examples:
            examples = [join_dicts(src, tgt)
                        for src, tgt in zip(src_examples, tgt_examples)]
        else:
            examples = src_examples
        for i, example in enumerate(examples):
            example["indices"] = i

        if opt is None or opt.dynamic_dict:
            for example in examples:
                src = example["src"]
                src_vocab = torchtext.vocab.Vocab(Counter(src))
                self.src_vocabs.append(src_vocab)
                # mapping source tokens to indices in the dynamic dict
                src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])

                self.src_vocabs.append(src_vocab)
                example["src_map"] = src_map

                if "tgt" in example:
                    tgt = example["tgt"]
                    mask = torch.LongTensor(
                            [0] + [src_vocab.stoi[w] for w in tgt] + [0])
                    example["alignment"] = mask

        keys = examples[0].keys()
        fields = [(k, fields[k]) for k in keys]
        examples = [torchtext.data.Example.fromlist([ex[k] for k in keys],
                                                    fields)
                    for ex in examples]

        def filter_pred(example):
            return 0 < len(example.src) <= opt.src_seq_length \
                and 0 < len(example.tgt) <= opt.tgt_seq_length

        super(ONMTDataset, self).__init__(examples, fields,
                                          filter_pred if opt is not None
                                          else None)

    @staticmethod
    def get_fields(nFeatures=0):
        fields = {}
        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)

        # fields = [("src_img", torchtext.data.Field(
        #     include_lengths=True))]

        for j in range(nFeatures):
            fields["src_feat_"+str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        # Added datum weight field
        fields["dw"] = torchtext.data.Field()

        def make_src(data, _):
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(src_size, len(data), src_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=make_src, sequential=False)

        def make_tgt(data, _):
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[:sent.size(0), i] = sent
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            postprocessing=make_tgt, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            sequential=False)

        return fields

    @staticmethod
    def load_fields(vocab):
        vocab = dict(vocab)
        fields = ONMTWeightedDataset.get_fields(
            len(ONMTWeightedDataset.collect_features(vocab)))
        for k, v in vocab.items():
            # Hack. Can't pickle defaultdict :(
            v.stoi = defaultdict(lambda: 0, v.stoi)
            fields[k].vocab = v
        return fields


def preprocess_args(parser):
    """
    replicate the arg parser on the module scope of IO
    """
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
                destination.write(str(random.random())+ "\n")
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

    parser.add_argument('-train_dw', default=None,
                        help="Path to the training datum-weight data")
    parser.add_argument('-valid_dw', default=None,
                        help="Path to the validation datum-weight data")

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)


    print('Preparing training ...')
    with codecs.open(opt.train_src, "r", "utf-8") as src_file:
        src_line = src_file.readline().strip().split()
        _, _, nFeatures = onmt.IO.extract_features(src_line)

    fields = ONMTWeightedDataset.get_fields(nFeatures)
    print("Building Training...")
    train = ONMTWeightedDataset(opt.train_src, opt.train_tgt, fields, opt, dw_path=opt.train_dw)
    print("Building Vocab...")
    ONMTWeightedDataset.build_vocab(train, opt)
    print(train)

    print("Building Valid...")
    valid = ONMTWeightedDataset(opt.valid_src, opt.valid_tgt, fields, opt, dw_path=opt.valid_dw)
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
    # weight_train_file, weight_val_file = create_dw_datasets()
    # train_path = test_create_datasets(weight_train_file, weight_val_file)
    train_path = "data_tests"
    test_load_datasets(train_path)