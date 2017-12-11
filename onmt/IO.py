# -*- coding: utf-8 -*-

import codecs
from collections import Counter, defaultdict
from itertools import chain, count

import torch
import torchtext.data
import torchtext.vocab

PAD_WORD = '<blank>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def __getstate__(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def __setstate__(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = __getstate__
torchtext.vocab.Vocab.__setstate__ = __setstate__


def load_fields(vocab):
    vocab = dict(vocab)
    n_src_features = len(collect_features(vocab, 'src'))
    n_tgt_features = len(collect_features(vocab, 'tgt'))
    fields = get_fields(n_src_features, n_tgt_features)
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
    return fields


def collect_features(fields, side="src"):
    assert side in ["src", "tgt"]
    feats = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feats.append(key)
    return feats


def extract_features(tokens):
    """
    tokens: A list of tokens, where each token consists of a word,
            optionally followed by u"￨"-delimited features.
    returns: A sequence of words, A sequence of features, and
    """
    if not tokens:
        return [], [], -1
    split_tokens = [token.split(u"￨") for token in tokens]
    split_tokens = [token for token in split_tokens if token[0]]
    token_size = len(split_tokens[0])
    assert all(len(token) == token_size for token in split_tokens), \
        "all words must have the same number of features"
    words_and_features = list(zip(*split_tokens))
    words = words_and_features[0]
    features = words_and_features[1:]
    return words, features, token_size - 1


def read_corpus_file(path, truncate, side):
    """
    path: location of a src or tgt file
    truncate: maximum sequence length (0 for unlimited)
    yields: (word, features, nfeat) triples for each line
    """
    with codecs.open(path, "r", "utf-8") as corpus_file:
        for i, line in enumerate(corpus_file):
            line = line.split()
            if truncate:
                line = line[:truncate]
            words, feats, n_feats = extract_features(line)
            example_dict = {side: words, "indices": i}
            if feats:
                prefix = side + "_feat_"
                example_dict.update((prefix + str(j), f)
                                    for j, f in enumerate(feats))
            yield example_dict, n_feats


def merge_vocabs(vocabs, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return torchtext.vocab.Vocab(merged,
                                 specials=[PAD_WORD, BOS_WORD, EOS_WORD],
                                 max_size=vocab_size)


def make_features(batch, side):
    """
    Args:
        batch (Variable): a batch of source or target data.
        side (str): for source or for target.
    Returns:
        A sequence of src/tgt tensors with optional feature tensors
        of size (len x batch).
    """
    assert side in ['src', 'tgt']
    if isinstance(batch.__dict__[side], tuple):
        data = batch.__dict__[side][0]
    else:
        data = batch.__dict__[side]
    feat_start = side + "_feat_"
    keys = sorted([k for k in batch.__dict__ if feat_start in k])
    features = [batch.__dict__[k] for k in keys]
    levels = [data] + features
    return torch.cat([level.unsqueeze(2) for level in levels], 2)


def save_vocab(fields):
    vocab = []
    for k, f in fields.items():
        if 'vocab' in f.__dict__:
            f.vocab.stoi = dict(f.vocab.stoi)
            vocab.append((k, f.vocab))
    return vocab


def collect_feature_dicts(fields, side):
    assert side in ['src', 'tgt']
    feature_dicts = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feature_dicts.append(fields[key].vocab)
    return feature_dicts


def get_fields(n_src_features, n_tgt_features):
    """
    n_src_features: the number of source features to create Field objects for.
    n_tgt_features: the number of target features to create Field objects for.
    returns: A dictionary whose keys are strings and whose values are the
            corresponding Field objects.
    """
    fields = {}
    fields["src"] = torchtext.data.Field(
        pad_token=PAD_WORD,
        include_lengths=True)

    # fields = [("src_img", torchtext.data.Field(
    #     include_lengths=True))]

    for j in range(n_src_features):
        fields["src_feat_"+str(j)] = \
            torchtext.data.Field(pad_token=PAD_WORD)

    fields["tgt"] = torchtext.data.Field(
        init_token=BOS_WORD, eos_token=EOS_WORD,
        pad_token=PAD_WORD)

    for j in range(n_tgt_features):
        fields["tgt_feat_"+str(j)] = \
            torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD,
                                 pad_token=PAD_WORD)

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


def build_vocab(train, opt):
    """
    train: an ONMTDataset
    """
    fields = train.fields
    fields["src"].build_vocab(train, max_size=opt.src_vocab_size,
                              min_freq=opt.src_words_min_frequency)
    for j in range(train.n_src_feats):
        fields["src_feat_" + str(j)].build_vocab(train)
    fields["tgt"].build_vocab(train, max_size=opt.tgt_vocab_size,
                              min_freq=opt.tgt_words_min_frequency)
    for j in range(train.n_tgt_feats):
        fields["tgt_feat_" + str(j)].build_vocab(train)

    # Merge the input and output vocabularies.
    if opt.share_vocab:
        # `tgt_vocab_size` is ignored when sharing vocabularies
        merged_vocab = merge_vocabs(
            [fields["src"].vocab, fields["tgt"].vocab],
            vocab_size=opt.src_vocab_size)
        fields["src"].vocab = merged_vocab
        fields["tgt"].vocab = merged_vocab


def join_dicts(*args):
    """
    args: dictionaries with disjoint keys
    returns: a single dictionary that has the union of these keys
    """
    return dict(chain(*[d.items() for d in args]))


def peek(seq):
    """
    sequence: an iterator
    returns: the first thing returned by calling next() on the iterator
        and an iterator created by re-chaining that value to the beginning
        of the iterator.
    """
    first = next(seq)
    return first, chain([first], seq)


class OrderedIterator(torchtext.data.Iterator):
    def create_batches(self):
        if self.train:
            self.batches = torchtext.data.pool(
                self.data(), self.batch_size,
                self.sort_key, self.batch_size_fn,
                random_shuffler=self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class ONMTDataset(torchtext.data.Dataset):
    """
    Defines a dataset for machine translation.
    An ONMTDataset is a collection that supports iteration over its
    examples. The parent class supports indexing as well, but future
    developments here may make that difficult (lazy iteration over
    examples because of large datasets, for example).
    """

    @staticmethod
    def sort_key(ex):
        "Sort in reverse size order"
        return -len(ex.src)

    def __init__(self, src_path, tgt_path, fields,
                 src_seq_length=0, tgt_seq_length=0,
                 src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                 use_filter_pred=True, dynamic_dict=True,
                 src_img_dir=None, **kwargs):
        """
        Create a translation dataset given paths and fields.

        src_path: location of source-side data
        tgt_path: location of target-side data or None. If should be the
            same length as the source-side data if it exists, but
            at present this is not checked.
        fields: a dictionary. keys are things like 'src', 'tgt', 'src_map',
            and 'alignment'
        src_img_dir: raises an error if not None because images are not
            supported yet.

        Initializes an ONMTDataset object with the following attributes:
        self.examples (might be a generator, might be a list, hard to say):
            A sequence of torchtext Example objects.
        self.fields (dict):
            A dictionary associating str keys with Field objects. Does not
            necessarily have the same keys as the input fields.

        A dataset basically supports iteration over all the examples it
        contains.
        """

        assert src_img_dir is None, "img data is not finished"

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []
        src_truncate = src_seq_length_trunc

        src_examples = read_corpus_file(src_path, src_truncate, "src")
        (_, src_feats), src_examples = peek(src_examples)
        src_examples = (ex for ex, nfeats in src_examples)
        self.n_src_feats = src_feats

        # if tgt_path exists, then we need to do the same thing as we did
        # for the source data
        if tgt_path is not None:
            tgt_truncate = tgt_seq_length_trunc
            tgt_examples = read_corpus_file(tgt_path, tgt_truncate, "tgt")
            (_, tgt_feats), tgt_examples = peek(tgt_examples)
            tgt_examples = (ex for ex, nfeats in tgt_examples)
            self.n_tgt_feats = tgt_feats
        else:
            self.n_tgt_feats = 0
            tgt_examples = None

        # examples: one for each src line or (src, tgt) line pair.
        # Each element is a dictionary whose keys represent at minimum
        # the src tokens and their indices and potentially also the
        # src and tgt features and alignment information.
        if tgt_examples is not None:
            examples = (join_dicts(src, tgt)
                        for src, tgt in zip(src_examples, tgt_examples))
        else:
            examples = src_examples

        if dynamic_dict:
            examples = self.dynamic_dict(examples)

        # Peek at the first to see which fields are used.
        ex, examples = peek(examples)
        keys = ex.keys()

        fields = [(k, fields[k]) for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples)
        out_examples = (torchtext.data.Example.fromlist(ex_values, fields)
                        for ex_values in example_values)

        def filter_pred(example):
            return 0 < len(example.src) <= src_seq_length \
                and 0 < len(example.tgt) <= tgt_seq_length

        super(ONMTDataset, self).__init__(
            out_examples,
            fields,
            filter_pred if use_filter_pred else None
        )

    def dynamic_dict(self, examples):
        for example in examples:
            src = example["src"]
            src_vocab = torchtext.vocab.Vocab(Counter(src))
            self.src_vocabs.append(src_vocab)
            # mapping source tokens to indices in the dynamic dict
            src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
            example["src_map"] = src_map

            if "tgt" in example:
                tgt = example["tgt"]
                mask = torch.LongTensor(
                        [0] + [src_vocab.stoi[w] for w in tgt] + [0])
                example["alignment"] = mask
            yield example

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(ONMTDataset, self).__reduce_ex__()

    def collapse_copy_scores(self, scores, batch, tgt_vocab):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            index = batch.indices.data[b]
            src_vocab = self.src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    scores[:, b, ti] += scores[:, b, offset + i]
                    scores[:, b, offset + i].fill_(1e-20)
        return scores


def load_image_libs():
    "Conditional import of torch image libs."
    global Image, transforms
    from PIL import Image
    from torchvision import transforms
