import torch
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example
from random import random


def bert_text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    return len(ex.tokens)


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


def create_sentence_instance(sentence, tokenizer,
                             max_seq_length, random_trunc=False):
    tokens = tokenizer.tokenize(sentence)
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 2
    if len(tokens) > max_num_tokens:
        if random_trunc is True:
            truncate_seq(tokens, max_num_tokens)
        else:
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


class BertDataset(TorchtextDataset):
    """Defines a BERT dataset composed of Examples along with its Fields.
    Args:
        fields_dict (dict[str, Field]): a dict containing all Field with
            its name.
        instances (Iterable[dict[]]): a list of dictionary, each dict
            represent one Example with its field specified by fields_dict.
    """

    def __init__(self, fields_dict, instances,
                 sort_key=bert_text_sort_key, filter_pred=None):
        self.sort_key = sort_key
        examples = []
        ex_fields = {k: [(k, v)] for k, v in fields_dict.items()}
        for instance in instances:
            ex = Example.fromdict(instance, ex_fields)
            examples.append(ex)
        fields_list = list(fields_dict.items())

        super(BertDataset, self).__init__(examples, fields_list, filter_pred)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)


class ClassifierDataset(BertDataset):
    """Defines a BERT dataset composed of Examples along with its Fields.
    Args:
        fields_dict (dict[str, Field]): a dict containing all Field with
            its name.
        data (list[]): a list of sequence (sentence or sentence pair),
            possible with its label becoming tuple(list[]).
    """

    def __init__(self, fields_dict, data, tokenizer,
                 max_seq_len=256, delimiter=' ||| '):
        if isinstance(data, tuple) is False:
            data = data, [None for _ in range(len(data))]
        instances = self.create_instances(
            data, tokenizer, delimiter, max_seq_len)
        super(ClassifierDataset, self).__init__(fields_dict, instances)

    def create_instances(self, data, tokenizer,
                         delimiter, max_seq_len):
        instances = []
        for sentence, label in zip(*data):
            sentences = sentence.strip().split(delimiter, 1)
            if len(sentences) == 2:
                sent_a, sent_b = sentences
                tokens, segment_ids = create_sentence_pair_instance(
                    sent_a, sent_b, tokenizer, max_seq_len)
            else:
                sentence = sentences[0]
                tokens, segment_ids = create_sentence_instance(
                    sentence, tokenizer, max_seq_len, random_trunc=False)
            instance = {
                "tokens": tokens,
                "segment_ids": segment_ids,
                "category": label}
            instances.append(instance)
        return instances


class TaggerDataset(BertDataset):
    """Defines a BERT dataset composed of Examples along with its Fields.
    Args:
        fields_dict (dict[str, Field]): a dict containing all Field with
            its name.
        data (list[]|tuple(list[])): a list of sequence, each sequence is
            composed with tokens that to be tagging. Can also combined with
            its tags as tuple([tokens], [tags])
    """

    def __init__(self, fields_dict, data, tokenizer,
                 max_seq_len=256, delimiter=' '):
        targer_field = fields_dict["token_labels"]
        self.pad_tok = targer_field.pad_token
        if hasattr(targer_field, 'vocab'):  # when predicting
            self.predict_tok = targer_field.vocab.itos[-1]
        if isinstance(data, tuple) is False:
            data = (data, [None for _ in range(len(data))])
        instances = self.create_instances(
            data, tokenizer, delimiter, max_seq_len)
        super(TaggerDataset, self).__init__(fields_dict, instances)

    def create_instances(self, datas, tokenizer, delimiter, max_seq_len):
        instances = []
        for words, taggings in zip(*datas):
            if isinstance(words, str):  # build from raw sentence
                words = words.strip().split(delimiter)
            if taggings is None:  # when predicting
                assert hasattr(self, 'predict_tok')
                taggings = [self.predict_tok for _ in range(len(words))]
            sentence = []
            tags = []
            max_num_tokens = max_seq_len - 2
            for word, tag in zip(words, taggings):
                tokens = tokenizer.tokenize(word)
                n_pad = len(tokens) - 1
                paded_tag = [tag] + [self.pad_tok] * n_pad
                if len(sentence) + len(tokens) > max_num_tokens:
                    break
                else:
                    sentence.extend(tokens)
                    tags.extend(paded_tag)
            sentence = ["[CLS]"] + sentence + ["[SEP]"]
            tags = [self.pad_tok] + tags + [self.pad_tok]
            segment_ids = [0 for _ in range(len(sentence))]
            instance = {
                "tokens": sentence,
                "segment_ids": segment_ids,
                "token_labels": tags}
            instances.append(instance)
        return instances
