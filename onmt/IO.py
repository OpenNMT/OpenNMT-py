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


def extract_features(tokens):
    "Given a list of token separate out words and features (if any)."

    split_tokens = [token.split(u"￨") for token in tokens]
    split_tokens = [token for token in split_tokens if token[0]]
    token_size = len(split_tokens[0])
    assert all(len(token) == token_size for token in split_tokens), \
        "all words must have the same number of features"
    words_and_features = list(zip(*split_tokens))
    words = words_and_features[0]
    features = words_and_features[1:]
    return words, features, token_size - 1


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
    merged = Counter(chain(*[vocab.freqs for vocab in vocabs]))
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
    features = sorted(batch.__dict__[k]
                      for k in batch.__dict__ if feat_start in k)
    levels = [data] + features
    return torch.cat([level.unsqueeze(2) for level in levels], 2)


def join_dicts(*args):
    """
    args: dictionaries with disjoint keys
    returns: a single dictionary that has the union of these keys
    """
    return dict(chain(*[d.items() for d in args]))


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
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        "Sort in reverse size order"
        return -len(ex.src)

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
        dw_path: location of datum-weights or None. If it exists, it
                  must be the same length as source and target.
        """
        if src_img_dir:
            self.type_ = "img"
        else:
            self.type_ = "text"

        if self.type_ == "text":
            self.src_vocabs = []
            src_truncate = 0 if opt is None else opt.src_seq_length_trunc
            src_point = next(self._read_corpus_file(src_path, src_truncate))
            self.nfeatures = src_point[2]
            src_data = self._read_corpus_file(src_path, src_truncate)
            src_examples = self._construct_examples(src_data, "src")
        else:
            # TODO finish this.
            if not transforms:
                load_image_libs()

        if tgt_path is not None:
            tgt_truncate = 0 if opt is None else opt.tgt_seq_length_trunc
            tgt_data = self._read_corpus_file(tgt_path, tgt_truncate)
            # assert len(src_data) == len(tgt_data), \
            #     "Len src and tgt do not match"
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

            # assert len(src_data) == len(dw_data), \
            #     "Len src and dw do not match"
            dw_examples = [{"dw": dw} for dw in dw_data]
        else:
            dw_examples = None

        # examples: one for each src line or (src, tgt) line pair.
        # Each element is a dictionary whose keys represent at minimum
        # the src tokens and their indices and potentially also the
        # src and tgt features and alignment information.
        if tgt_examples and dw_path:
            examples = (join_dicts(src, tgt, dw)
                        for src, tgt, dw in zip(src_examples,
                                                tgt_examples,
                                                dw_examples))
        elif tgt_examples:
            examples = (join_dicts(src, tgt)
                        for src, tgt in zip(src_examples, tgt_examples))
        else:
            examples = src_examples

        def dynamic_dict(examples):
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

        if opt is None or opt.dynamic_dict:
            examples = dynamic_dict(examples)

        # Peek at the first to see which fields are used.
        ex = next(examples)
        keys = ex.keys()
        fields = [(k, fields[k])
                  for k in (list(keys) + ["indices"])]

        def construct_final(examples):
            for i, ex in enumerate(examples):
                yield torchtext.data.Example.fromlist(
                    [ex[k] for k in keys] + [i],
                    fields)

        def filter_pred(example):
            return 0 < len(example.src) <= opt.src_seq_length \
                and 0 < len(example.tgt) <= opt.tgt_seq_length

        super(ONMTDataset, self).__init__(
            construct_final(chain([ex], examples)),
            fields,
            filter_pred if opt is not None
            else None)

    def _read_corpus_file(self, path, truncate):
        """
        path: location of a src or tgt file
        truncate: maximum sequence length (0 for unlimited)

        returns: (word, features, nfeat) triples for each line
        """
        with codecs.open(path, "r", "utf-8") as corpus_file:
            lines = (line.split() for line in corpus_file)
            if truncate:
                lines = (line[:truncate] for line in lines)
            for line in lines:
                yield extract_features(line)

    def _construct_examples(self, lines, side):
        assert side in ["src", "tgt"]
        for line in lines:
            words, feats, _ = line
            example_dict = {side: words}
            if feats:
                prefix = side + "_feat_"
                example_dict.update((prefix + str(j), f)
                                    for j, f in enumerate(feats))
            yield example_dict

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(ONMTDataset, self).__reduce_ex__()

    def collapse_copy_scores(self, scores, batch, tgt_vocab):
        """Given scores from an expanded dictionary
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

    @staticmethod
    def load_fields(vocab):
        vocab = dict(vocab)
        fields = ONMTDataset.get_fields(
            len(ONMTDataset.collect_features(vocab)))
        for k, v in vocab.items():
            # Hack. Can't pickle defaultdict :(
            v.stoi = defaultdict(lambda: 0, v.stoi)
            fields[k].vocab = v
        return fields

    @staticmethod
    def save_vocab(fields):
        vocab = []
        for k, f in fields.items():
            if 'vocab' in f.__dict__:
                f.vocab.stoi = dict(f.vocab.stoi)
                vocab.append((k, f.vocab))
        return vocab

    @staticmethod
    def collect_features(fields, side="src"):
        assert side in ["src", "tgt"]
        feats = []
        for j in count():
            key = side + "_feat_" + str(j)
            if key not in fields:
                break
            feats.append(key)
        return feats

    @staticmethod
    def collect_feature_dicts(fields):
        feature_dicts = []
        for j in count():
            key = "src_feat_" + str(j)
            if key not in fields:
                break
            feature_dicts.append(fields[key].vocab)
        return feature_dicts

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
        fields["dw"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            sequential=False)

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
    def build_vocab(train, opt):
        fields = train.fields
        fields["src"].build_vocab(train, max_size=opt.src_vocab_size,
                                  min_freq=opt.src_words_min_frequency)
        for j in range(train.nfeatures):
            fields["src_feat_" + str(j)].build_vocab(train)
        fields["tgt"].build_vocab(train, max_size=opt.tgt_vocab_size,
                                  min_freq=opt.tgt_words_min_frequency)

        # Merge the input and output vocabularies.
        if opt.share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            merged_vocab = merge_vocabs(
                [fields["src"].vocab, fields["tgt"].vocab],
                vocab_size=opt.src_vocab_size)
            fields["src"].vocab = merged_vocab
            fields["tgt"].vocab = merged_vocab


def load_image_libs():
    "Conditional import of torch image libs."
    global Image, transforms
    from PIL import Image
    from torchvision import transforms
