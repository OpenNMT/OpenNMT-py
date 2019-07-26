from argparse import ArgumentParser
from tqdm import tqdm
import csv
from random import random
from onmt.utils.bert_tokenization import BertTokenizer, \
    PRETRAINED_VOCAB_ARCHIVE_MAP
import json
from onmt.inputters.inputter import get_bert_fields, \
    _build_bert_fields_vocab
from onmt.inputters.dataset_bert import BertDataset
from collections import Counter, defaultdict
import torch
import os


def truncate_seq(tokens, max_num_tokens):
    """Truncates a sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens)
        if total_length <= max_num_tokens:
            break
        assert len(tokens) >= 1
        # We want to sometimes truncate from the front and sometimes
        # from the back to add more randomness and avoid biases.
        if random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length.
       Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def create_sentence_instance(sentence, tokenizer, max_seq_length):
    tokens = tokenizer.tokenize(sentence)
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 2
    if len(tokens) > max_num_tokens:
        tokens = tokens[:max_num_tokens]
    tokens_processed = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0 for _ in range(len(tokens) + 2)]
    return tokens_processed, segment_ids


def create_sentence_pair_instance(sent_a, sent_b, tokenizer, max_seq_length):
    tokens_a = tokenizer.tokenize(sent_a)
    tokens_b = tokenizer.tokenize(sent_b)
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3
    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
    tokens_processed = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
    segment_ids = [0 for _ in range(len(tokens_a) + 2)] + \
                  [1 for _ in range(len(tokens_b) + 1)]
    return tokens_processed, segment_ids


def create_instances(records, skip_head, tokenizer, max_seq_length,
                     column_a, column_b, label_column, labels):
    instances = []
    for _i, record in tqdm(enumerate(records), desc="Process", unit=" lines"):
        if _i == 0 and skip_head:
            continue
        else:
            sentence_a = record[column_a].strip()
            if column_b is not None:
                sentence_b = record[column_b].strip()
            else:
                sentence_b = None
            if label_column is not None:
                label = record[label_column].strip()
                target = None
                for i, label_name in enumerate(labels):
                    if label == label_name:
                        target = i
                if target is None:
                    raise ValueError("Unregconizable label: %s" % label)
            else:
                target = -1
        if column_b is None:
            tokens_processed, segment_ids = create_sentence_instance(
                sentence_a, tokenizer, max_seq_length)
        else:
            tokens_processed, segment_ids = create_sentence_pair_instance(
                sentence_a, sentence_b, tokenizer, max_seq_length)
        instance = {
            "tokens": tokens_processed,
            "segment_ids": segment_ids,
            "category": target}
        instances.append(instance)
    return instances


def build_instances_from_csv(data, skip_head, tokenizer, input_columns,
                             label_column, labels, max_seq_len):
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
        instances = create_instances(lines, skip_head, tokenizer, max_seq_len,
                                     column_a, column_b, label_column, labels)
    return instances


def _build_bert_vocab(vocab, name, counters, min_freq=0):
    """ similar to _load_vocab in inputter.py, but build from a vocab list.
        in place change counters
    """
    vocab_size = len(vocab)
    for i, token in enumerate(vocab):
        counters[name][token] = vocab_size - i + min_freq
    return vocab, vocab_size


def build_vocab_from_tokenizer(fields, tokenizer, tokens_min_frequency=0,
                               vocab_size_multiple=1):
    vocab_list = list(tokenizer.vocab.keys())
    counters = defaultdict(Counter)
    _, vocab_size = _build_bert_vocab(vocab_list, "tokens", counters)
    fields_vocab = _build_bert_fields_vocab(fields, counters, vocab_size,
                                            tokens_min_frequency,
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

    assert os.path.isfile(opts.data), "Please check path of %s" % opts.data

    if opts.data_type == "csv":
        assert len(opts.input_columns) in [1, 2],\
            "Please indicate N.colomn for sentence A (and B)"


def main():
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default=None, required=True,
                        help="input data to prepare: path/filename.suffix")
    parser.add_argument('--data_type', type=str, default="csv",
                        help="input data type")
    parser.add_argument('--skip_head', action="store_true",
                        help="If csv file contain head line.")

    parser.add_argument('--input_columns', nargs='+', default=[None],
                        help="Column numbers where contain sentence A(,B)")
    parser.add_argument('--label_column', type=int, default=None,
                        help="Column number where contain label")
    parser.add_argument('--labels', nargs='+', default=[None],
                        help="labels of sentence")

    parser.add_argument('--task', type=str, default="classification",
                        choices=["classification", "generation"],
                        help="Target task to perform")
    parser.add_argument("--corpus_type", type=str, default="train",
                        choices=["train", "valid", "test"])
    parser.add_argument('--save_data', '-save_data', type=str,
                        default=None, required=True,
                        help="Output file Prefix for the prepared data")
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

    args = parser.parse_args()
    validate_preprocess_bert_opts(args)

    print("Load data file %s with skip head %s" % (args.data, args.skip_head))
    input_columns = args.input_columns
    label_column = args.label_column
    print("Input column at {}, label'{}'".format(input_columns, label_column))
    print("Task: '%s', model: '%s', corpus: '%s'."
          % (args.task, args.bert_model, args.corpus_type))

    fields = get_bert_fields(args.task)
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    # Build instances from csv file
    if args.data_type == 'csv':
        instances = build_instances_from_csv(
            args.data, args.skip_head, tokenizer,
            input_columns, label_column, args.labels, args.max_seq_len)
    else:
        raise NotImplementedError("Not support other file type yet!")

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
        fields_vocab = build_vocab_from_tokenizer(fields, tokenizer,
                                                  args.tokens_min_frequency,
                                                  args.vocab_size_multiple)
        bert_vocab_file = args.save_data + ".vocab.pt"
        torch.save(fields_vocab, bert_vocab_file)


if __name__ == '__main__':
    main()
