#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data files and build vocabulary for Bert model.
"""
from onmt.utils.parse import ArgumentParser
from tqdm import tqdm
import csv
from collections import Counter, defaultdict
import torch
import codecs
from random import shuffle
from onmt.utils.bert_tokenization import BertTokenizer
from onmt.inputters.inputter import get_bert_fields, \
    _build_bert_fields_vocab
import onmt.opts as opts
from onmt.inputters.dataset_bert import ClassifierDataset, \
     TaggerDataset
from onmt.utils.logging import init_logger, logger


def shuffle_pair_list(list_a, list_b):
    assert len(list_a) == len(list_b),\
        "Two list to shuffle should be equal length"
    logger.info("Shuffle all instance")
    pair_list = list(zip(list_a, list_b))
    shuffle(pair_list)
    list_a, list_b = zip(*pair_list)
    return list_a, list_b


def build_label_vocab_from_path(paths):
    labels = []
    for filename in paths:
        label = filename.split('/')[-2]
        if label not in labels:
            labels.append(label)
    return labels


def _build_bert_vocab(vocab, name, counters):
    """ similar to _load_vocab in inputter.py, but build from a vocab list.
        in place change counters
    """
    vocab_size = len(vocab)
    for i, token in enumerate(vocab):
        counters[name][token] = vocab_size - i
    return vocab, vocab_size


def build_vocab_from_tokenizer(fields, tokenizer, named_labels):
    logger.info("Building token vocab from BertTokenizer...")
    vocab_list = list(tokenizer.vocab.keys())
    counters = defaultdict(Counter)
    _, vocab_size = _build_bert_vocab(vocab_list, "tokens", counters)

    label_name, label_list = named_labels
    logger.info("Building label vocab {}...".format(named_labels))
    _, _ = _build_bert_vocab(label_list, label_name, counters)

    fields_vocab = _build_bert_fields_vocab(fields, counters, vocab_size,
                                            label_name)
    return fields_vocab


def build_save_vocab(fields, tokenizer, label_vocab, opt):
    if opt.sort_label_vocab is True:
        label_vocab.sort()
    if opt.task == "classification":
        named_labels = ("category", label_vocab)
    if opt.task == "tagging":
        named_labels = ("token_labels", label_vocab)

    fields_vocab = build_vocab_from_tokenizer(
        fields, tokenizer, named_labels)
    bert_vocab_file = opt.save_data + ".vocab.pt"
    torch.save(fields_vocab, bert_vocab_file)


def create_cls_instances_from_csv(opt):
    logger.info("Reading csv with input in column %s, label in column %s"
                % (opt.input_columns, opt.label_column))
    with codecs.open(opt.data, "r", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile, delimiter=opt.delimiter, quotechar=None)
        lines = list(reader)
        if opt.skip_head is True:
            lines = lines[1:]
        if len(opt.input_columns) == 1:
            column_a = int(opt.input_columns[0])
            column_b = None
        else:
            column_a = int(opt.input_columns[0])
            column_b = int(opt.input_columns[1])

        instances, labels, label_vocab = [], [], opt.labels
        for line in tqdm(lines, desc="Process", unit=" lines"):
            label = line[opt.label_column].strip()
            if label not in label_vocab:
                label_vocab.append(label)
            sentence = line[column_a].strip()
            if column_b is not None:
                sentence_b = line[column_b].strip()
                sentence = sentence + ' ||| ' + sentence_b
            instances.append(sentence)
            labels.append(label)
        logger.info("total %d line loaded with skip_head [%s]"
                    % (len(lines), opt.skip_head))

    return instances, labels, label_vocab


def create_cls_instances_from_files(opt):
    instances = []
    labels = []
    label_vocab = build_label_vocab_from_path(opt.data)
    for filename in opt.data:
        label = filename.split('/')[-2]
        with codecs.open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print("total {} line of File {} loaded for label: {}.".format(
                len(lines), filename, label))
            lines_labels = [label for _ in range(len(lines))]
            instances.extend(lines)
            labels.extend(lines_labels)
    return instances, labels, label_vocab


def build_cls_dataset(corpus_type, fields, tokenizer, opt):
    """Build classification dataset with vocab file if train set"""
    assert corpus_type in ['train', 'valid']
    if opt.file_type == 'csv':
        instances, labels, label_vocab = create_cls_instances_from_csv(opt)
    else:
        instances, labels, label_vocab = create_cls_instances_from_files(opt)
    logger.info("Exiting labels:%s" % label_vocab)
    if corpus_type == 'train':
        build_save_vocab(fields, tokenizer, label_vocab, opt)

    if opt.do_shuffle is True:
        instances, labels = shuffle_pair_list(instances, labels)
    cls_instances = instances, labels
    logger.info("Building %s dataset..." % corpus_type)
    dataset = ClassifierDataset(
            fields, cls_instances, tokenizer, opt.max_seq_len)
    return dataset, len(cls_instances[0])


def create_tag_instances_from_file(opt):
    logger.info("Reading tag with token in column %s, tag in column %s"
                % (opt.input_columns, opt.label_column))
    sentences, taggings = [], []
    tag_vocab = opt.labels
    with codecs.open(opt.data, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print("total {} line of file {} loaded.".format(
            len(lines), opt.data))
        sentence_sofar = []
        for line in tqdm(lines, desc="Process", unit=" lines"):
            line = line.strip()
            if line is '':
                if len(sentence_sofar) > 0:
                    tokens, tags = zip(*sentence_sofar)
                    sentences.append(tokens)
                    taggings.append(tags)
                sentence_sofar = []
            else:
                elements = line.split(opt.delimiter)
                token = elements[opt.input_columns]
                tag = elements[opt.label_column]
                if tag not in tag_vocab:
                    tag_vocab.append(tag)
                sentence_sofar.append((token, tag))
        print("total {} sentence loaded.".format(len(sentences)))
        print("All tags:", tag_vocab)

    return sentences, taggings, tag_vocab


def build_tag_dataset(corpus_type, fields, tokenizer, opt):
    """Build tagging dataset with vocab file if train set"""
    assert corpus_type in ['train', 'valid']
    sentences, taggings, tag_vocab = create_tag_instances_from_file(opt)
    logger.info("Exiting Tags:%s" % tag_vocab)
    if corpus_type == 'train':
        build_save_vocab(fields, tokenizer, tag_vocab, opt)

    if opt.do_shuffle is True:
        sentences, taggings = shuffle_pair_list(sentences, taggings)

    tag_instances = sentences, taggings
    logger.info("Building %s dataset..." % corpus_type)
    dataset = TaggerDataset(
            fields, tag_instances, tokenizer, opt.max_seq_len)
    return dataset, len(tag_instances[0])


def _get_parser():
    parser = ArgumentParser(description='preprocess_bert.py')
    opts.config_opts(parser)
    opts.preprocess_bert_opts(parser)
    return parser


def main(opt):
    init_logger(opt.log_file)
    opt = ArgumentParser.validate_preprocess_bert_opts(opt)
    logger.info("Preprocess dataset...")

    fields = get_bert_fields(opt.task)
    logger.info("Get fields for Task: '%s'." % opt.task)

    tokenizer = BertTokenizer.from_pretrained(
        opt.vocab_model, do_lower_case=opt.do_lower_case)
    logger.info("Use pretrained tokenizer: '%s', do_lower_case [%s]"
                % (opt.vocab_model, opt.do_lower_case))

    if opt.task == "classification":
        dataset, n_instance = build_cls_dataset(
            opt.corpus_type, fields, tokenizer, opt)

    elif opt.task == "tagging":
        dataset, n_instance = build_tag_dataset(
            opt.corpus_type, fields, tokenizer, opt)
    # Save processed data in OpenNMT format
    onmt_filename = opt.save_data + ".{}.0.pt".format(opt.corpus_type)
    dataset.save(onmt_filename)
    logger.info("* save num_example [%d], max_seq_len [%d] to [%s]."
                % (n_instance, opt.max_seq_len, onmt_filename))


if __name__ == '__main__':
    parser = _get_parser()
    opt = parser.parse_args()
    main(opt)
