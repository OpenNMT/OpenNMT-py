from argparse import ArgumentParser
from tqdm import tqdm
import csv
from random import shuffle
from onmt.utils.bert_tokenization import BertTokenizer, \
    PRETRAINED_VOCAB_ARCHIVE_MAP
import json
from onmt.inputters.inputter import get_bert_fields, \
    _build_bert_fields_vocab
from onmt.inputters.dataset_bert import BertDataset, \
    create_sentence_instance, create_sentence_pair_instance
from collections import Counter, defaultdict
import torch
import os
import codecs


def create_instances_from_csv(records, skip_head, tokenizer, max_seq_length,
                              column_a, column_b, label_column, labels):
    instances = []
    for _i, record in tqdm(enumerate(records), desc="Process", unit=" lines"):
        if _i == 0 and skip_head:
            continue
        else:
            sentence_a = record[column_a].strip()
            if column_b is None:
                tokens_processed, segment_ids = create_sentence_instance(
                    sentence_a, tokenizer, max_seq_length)
            else:
                sentence_b = record[column_b].strip()
                tokens_processed, segment_ids = create_sentence_pair_instance(
                    sentence_a, sentence_b, tokenizer, max_seq_length)

            label = record[label_column].strip()
            if label not in labels:
                labels.append(label)
            instance = {
                "tokens": tokens_processed,
                "segment_ids": segment_ids,
                "category": label}
            instances.append(instance)
    return instances, labels


def build_instances_from_csv(data, skip_head, tokenizer, input_columns,
                             label_column, labels, max_seq_len, do_shuffle):
    with open(data, "r", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar=None)
        lines = list(reader)
        print("total {} line loaded: ".format(len(lines)))
        if len(input_columns) == 1:
            column_a = int(input_columns[0])
            column_b = None
        else:
            column_a = int(input_columns[0])
            column_b = int(input_columns[1])
        instances, labels = create_instances_from_csv(
            lines, skip_head, tokenizer, max_seq_len,
            column_a, column_b, label_column, labels)
    if do_shuffle is True:
        print("Shuffle all {} instance".format(len(instances)))
        shuffle(instances)
    return instances, labels


def create_instances_from_file(records, label, tokenizer, max_seq_length):
    instances = []
    for _i, record in tqdm(enumerate(records), desc="Process", unit=" lines"):
        sentence = record.strip()
        tokens_processed, segment_ids = create_sentence_instance(
            sentence, tokenizer, max_seq_length, random_trunc=True)
        instance = {
            "tokens": tokens_processed,
            "segment_ids": segment_ids,
            "category": label}
        instances.append(instance)
    return instances


def build_instances_from_files(data, labels, tokenizer,
                               max_seq_len, do_shuffle):
    instances = []
    for filename in data:
        label = filename.split('/')[-2]
        with codecs.open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print("total {} line of File {} loaded for label: {}.".format(
                len(lines), filename, label))
            file_instances = create_instances_from_file(
                lines, label, tokenizer, max_seq_len)
            instances.extend(file_instances)
    if do_shuffle is True:
        print("Shuffle all {} instance".format(len(instances)))
        shuffle(instances)
    return instances


def create_tag_instance_from_sentence(token_pairs, tokenizer, max_seq_len,
                                      pad_tok):
    """
    token_pairs: list of (word, tag) pair that form a sentence
    tokenizer: tokenizer we use to tokenizer the words in token_pairs
    max_seq_len: max sequence length that a instance could contain
    """
    sentence = []
    tags = []
    max_num_tokens = max_seq_len - 2
    for (word, tag) in token_pairs:
        tokens = tokenizer.tokenize(word)
        n_pad = len(tokens) - 1
        paded_tag = [tag] + [pad_tok] * n_pad
        if len(sentence) + len(tokens) > max_num_tokens:
            break
        else:
            sentence.extend(tokens)
            tags.extend(paded_tag)
    sentence = ["[CLS]"] + sentence + ["[SEP]"]
    tags = [pad_tok] + tags + [pad_tok]
    segment_ids = [0 for _ in range(len(sentence))]
    instance = {
        "tokens": sentence,
        "segment_ids": segment_ids,
        "token_labels": tags
    }
    return instance


def build_tag_instances_from_file(filename, skip_head, tokenizer, max_seq_len,
                                  token_column, tag_column, tags, do_shuffle,
                                  pad_tok, delimiter=' '):
    sentences = []
    labels = [] if tags is None else tags
    with codecs.open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if skip_head is True:
            lines = lines[1:]
        print("total {} line of file {} loaded.".format(
            len(lines), filename))
        sentence_sofar = []
        for line in tqdm(lines, desc="Process", unit=" lines"):
            line = line.strip()
            if line is '':
                if len(sentence_sofar) > 0:
                    sentences.append(sentence_sofar)
                sentence_sofar = []
            else:
                elements = line.split(delimiter)
                token = elements[token_column]
                tag = elements[tag_column]
                if tag not in labels:
                    labels.append(tag)
                sentence_sofar.append((token, tag))
        print("total {} sentence loaded.".format(len(sentences)))
        print("All tags:", labels)

    instances = []
    for sentence in sentences:
        instance = create_tag_instance_from_sentence(
            sentence, tokenizer, max_seq_len, pad_tok)
        instances.append(instance)

    if do_shuffle is True:
        print("Shuffle all {} instance".format(len(instances)))
        shuffle(instances)
    return instances, labels


def _build_bert_vocab(vocab, name, counters):
    """ similar to _load_vocab in inputter.py, but build from a vocab list.
        in place change counters
    """
    vocab_size = len(vocab)
    for i, token in enumerate(vocab):
        counters[name][token] = vocab_size - i
    return vocab, vocab_size


def build_vocab_from_tokenizer(fields, tokenizer, named_labels,
                               tokens_min_frequency=0, vocab_size_multiple=1):
    vocab_list = list(tokenizer.vocab.keys())
    counters = defaultdict(Counter)
    _, vocab_size = _build_bert_vocab(vocab_list, "tokens", counters)

    if named_labels is not None:
        label_name, label_list = named_labels
        _, _ = _build_bert_vocab(label_list, label_name, counters)
    else:
        label_name = None

    fields_vocab = _build_bert_fields_vocab(fields, counters, vocab_size,
                                            label_name, tokens_min_frequency,
                                            vocab_size_multiple)
    return fields_vocab


def save_data_as_json(instances, json_name):
    instances_json = [json.dumps(instance) for instance in instances]
    num_instances = 0
    with open(json_name, 'w') as json_file:
        for instance in instances_json:
            json_file.write(instance + '\n')
            num_instances += 1
    return num_instances


def validate_preprocess_bert_opts(opts):
    assert opts.bert_model in PRETRAINED_VOCAB_ARCHIVE_MAP.keys(), \
        "Unsupported Pretrain model '%s'" % (opts.bert_model)
    for filename in opts.data:
        assert os.path.isfile(filename),\
             "Please check path of %s" % filename

    if args.task == "tagging":
        assert args.data_type == 'txt',\
            "For sequence tagging, only txt file is supported."

        assert len(opts.input_columns) == 1,\
            "For sequence tagging, only one column for input tokens."
        opts.input_columns = opts.input_columns[0]

        assert args.label_column is not None,\
            "For sequence tagging, label column should be given."

    if opts.data_type == "csv":
        assert len(opts.data) == 1,\
            "For csv, only one file is needed."
        assert len(opts.input_columns) in [1, 2],\
            "Please indicate N.colomn for sentence A (and B)"
        assert args.label_column is not None,\
            "For csv file, label column should be given."

        # if opts.label_column is not None:
        #     assert len(opts.labels) != 0,\
        #         "label list is needed when csv contain label column"

    # elif opts.data_type == "txt":
    #     if opts.task == "classification":
    #         assert len(opts.datas) == len(opts.labels), \
    #             "Label should correspond to input files"
    return opts


def _get_parser():
    parser = ArgumentParser(description='preprocess_bert.py')

    parser.add_argument('--data', type=str, nargs='+', default=[],
                        required=True, help="input datas to prepare: [CLS]" +
                        "Single file for csv with column indicate label," +
                        "One file for each class as path/label/file; [TAG]" +
                        "Single file contain (tok, tag) in each line," +
                        "Sentence separated by blank line.")
    parser.add_argument('--data_type', type=str, default="csv",
                        choices=["csv", "txt"],
                        help="input data type")
    parser.add_argument('--skip_head', action="store_true",
                        help="CSV: If csv file contain head line.")

    parser.add_argument('--input_columns', type=int, nargs='+', default=[],
                        help="CSV: Column where contain sentence A(,B)")
    parser.add_argument('--label_column', type=int, default=None,
                        help="CSV: Column where contain label")
    parser.add_argument('--labels', type=str, nargs='+', default=[],
                        help="Candidate labels. If not given, build from " +
                        "input file and sort in alphabetic order.")
    parser.add_argument('--delimiter', '-d', type=str, default=' ',
                        help="CSV: delimiter used for seperate column.")

    parser.add_argument('--task', type=str, default="classification",
                        choices=["classification", "tagging"],
                        help="Target task to perform")
    parser.add_argument("--corpus_type", type=str, default="train",
                        choices=["train", "valid", "test"])
    parser.add_argument('--save_data', '-save_data', type=str,
                        default=None, required=True,
                        help="Output file Prefix for the prepared data")
    parser.add_argument("--do_shuffle", action="store_true",
                        help='shuffle data')
    parser.add_argument("--bert_model", type=str,
                        default="bert-base-multilingual-uncased",
                        choices=["bert-base-uncased", "bert-large-uncased",
                                 "bert-base-cased", "bert-large-cased",
                                 "bert-base-multilingual-uncased",
                                 "bert-base-multilingual-cased",
                                 "bert-base-chinese"],
                        help="Bert pretrained model to finetuning with.")

    parser.add_argument("--do_lower_case", action="store_true",
                        help='lowercase data')
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="max sequence length for prepared data,"
                        "set the limite of position encoding")
    parser.add_argument("--tokens_min_frequency", type=int, default=0)
    parser.add_argument("--vocab_size_multiple", type=int, default=1)
    parser.add_argument("--save_json", action="store_true",
                        help='save a copy of data in json form.')
    return parser


def main(args):
    print("Task: '%s', model: '%s', corpus: '%s'."
          % (args.task, args.bert_model, args.corpus_type))

    fields = get_bert_fields(args.task)
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    if args.task == "classification":
        # Build instances from csv file
        if args.data_type == 'csv':
            filename = args.data[0]
            print("Load data file %s with skip head %s" % (
                filename, args.skip_head))
            input_columns = args.input_columns
            label_column = args.label_column
            print("Input column at {}, label at [{}]".format(
                input_columns, label_column))
            instances, labels = build_instances_from_csv(
                filename, args.skip_head, tokenizer,
                input_columns, label_column,
                args.labels, args.max_seq_len, args.do_shuffle)
            labels.sort()
            args.labels = labels
            print("Labels:", args.labels)
        elif args.data_type == 'txt':
            if len(args.labels) == 0:
                print("Build labels from file dir...")
                labels = []
                for filename in args.data:
                    label = filename.split('/')[-2]
                    if label not in labels:
                        labels.append(label)
                labels.sort()
                args.labels = labels
            print("Labels:", args.labels)
            instances = build_instances_from_files(
                args.data, args.labels, tokenizer,
                args.max_seq_len, args.do_shuffle)
        else:
            raise NotImplementedError("Not support other file type yet!")

    if args.task == "tagging":
        pad_tok = fields["token_labels"].pad_token  # "[PAD]" for Bert Paddings
        filename = args.data[0]
        print("Load data file %s with skip head %s" % (
            filename, args.skip_head))
        token_column, tag_column = args.input_columns, args.label_column
        instances, labels = build_tag_instances_from_file(
            filename, args.skip_head, tokenizer, args.max_seq_len,
            token_column, tag_column, args.labels, args.do_shuffle,
            pad_tok, delimiter=args.delimiter)
        labels.sort()
        args.labels = [pad_tok] + labels
        print("Labels:", args.labels)

    # Save processed data in OpenNMT format
    onmt_filename = args.save_data + ".{}.0.pt".format(args.corpus_type)
    # Build BertDataset from instances collected from different document
    dataset = BertDataset(fields, instances)
    dataset.save(onmt_filename)
    print("save processed data {}, num_example {}, max_seq_len {}".format(
            onmt_filename, len(instances), args.max_seq_len))

    if args.save_json:
        json_name = args.save_data + ".{}.json".format(args.corpus_type)
        num_instances = save_data_as_json(instances, json_name)
        print("output file {}, num_example {}, max_seq_len {}".format(
            json_name, num_instances, args.max_seq_len))

    # Build file Vocab.pt from tokenizer
    if args.corpus_type == "train":
        print("Generating vocab from corresponding text file...")
        if args.task == "classification":
            named_labels = ("category", args.labels)
        if args.task == "tagging":
            named_labels = ("token_labels", args.labels)
        print("Save Labels:", named_labels, "in vocab file.")

        fields_vocab = build_vocab_from_tokenizer(
            fields, tokenizer, named_labels,
            args.tokens_min_frequency, args.vocab_size_multiple)
        bert_vocab_file = args.save_data + ".vocab.pt"
        torch.save(fields_vocab, bert_vocab_file)


if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args()
    args = validate_preprocess_bert_opts(args)
    main(args)
