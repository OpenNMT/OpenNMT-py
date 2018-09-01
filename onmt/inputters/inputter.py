# -*- coding: utf-8 -*-

import glob
import os
import io
from collections import Counter, defaultdict
from itertools import count
from functools import partial

import torch
import torchtext.data
from torchtext.vocab import Vocab

from onmt.inputters.dataset_base import UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD
from onmt.inputters.text_dataset import TextDataset, extract_text_features
from onmt.inputters.image_dataset import ImageDataset
from onmt.inputters.audio_dataset import AudioDataset
from onmt.utils.logging import logger


def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


Vocab.__getstate__ = _getstate
Vocab.__setstate__ = _setstate


def filter_pred(ex, use_src_len, use_tgt_len, min_src_len, max_src_len,
                min_tgt_len, max_tgt_len):
    """
    A generalized function for filtering examples based on the length of their
    src or tgt values. Rather than being used by itself as the filter_pred
    argument to a dataset, it should be partially evaluated with everything
    specified except the value of the example.
    """
    return (not use_src_len or min_src_len < len(ex.src) <= max_src_len) and \
        (not use_tgt_len or min_tgt_len < len(ex.tgt) <= max_tgt_len)


# below: postprocessing functions for fields. What are they for?
def make_src_map(data, vocab):
    """ ? """
    src_size = max([t.size(0) for t in data])
    src_vocab_size = max([t.max() for t in data]) + 1
    alignment = torch.zeros(src_size, len(data), src_vocab_size)
    for i, sent in enumerate(data):
        for j, t in enumerate(sent):
            alignment[j, i, t] = 1
    return alignment


def make_tgt(data, vocab):
    """ ? """
    tgt_size = max([t.size(0) for t in data])
    alignment = torch.zeros(tgt_size, len(data)).long()
    for i, sent in enumerate(data):
        alignment[:sent.size(0), i] = sent
    return alignment


def make_img(data, vocab):
    """ ? """
    c = data[0].size(0)
    h = max([t.size(1) for t in data])
    w = max([t.size(2) for t in data])
    imgs = torch.ones(len(data), c, h, w)
    for i, img in enumerate(data):
        imgs[i, :, 0:img.size(1), 0:img.size(2)] = img
    return imgs


def make_audio(data, vocab):
    """ ? """
    nfft = data[0].size(0)
    t = max([t.size(1) for t in data])
    sounds = torch.zeros(len(data), 1, nfft, t)
    for i, spect in enumerate(data):
        sounds[i, :, :, 0:spect.size(1)] = spect
    return sounds


def get_fields(data_type, n_src_features, n_tgt_features, dynamic_dict=False):
    """
    Args:
        data_type: type of the source input. Options are [text|img|audio].
        n_src_features: the number of source features to
            create `torchtext.data.Field` for.
        n_tgt_features: the number of target features to
            create `torchtext.data.Field` for.
        dynamic_dict (bool): whether the model has a dynamic dict/copy attn.
            If so, additional fields are required.

    Returns:
        A dictionary whose keys are strings and whose values are the
        corresponding Field objects.
    """
    if data_type not in ['text', 'img', 'audio']:
        raise ValueError("Data type not implemented")

    fields = {}

    # at the moment, "data_type" only refers to the source
    if data_type == 'text':
        fields["src"] = torchtext.data.Field(
            tokenize=str.split, pad_token=PAD_WORD, include_lengths=True)
    elif data_type == 'img':
        fields["src"] = torchtext.data.Field(
            tokenize=str.split, use_vocab=False, dtype=torch.float,
            postprocessing=make_img, sequential=False)
    else:
        fields["src"] = torchtext.data.Field(
            tokenize=str.split, use_vocab=False, dtype=torch.float,
            postprocessing=make_audio, sequential=False)

    for j in range(n_src_features):
        fields["src_feat_" + str(j)] = torchtext.data.Field(
            pad_token=PAD_WORD, tokenize=str.split)

    fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD, tokenize=str.split)

    for j in range(n_tgt_features):
        fields["tgt_feat_" + str(j)] = \
            torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD,
                                 pad_token=PAD_WORD, tokenize=str.split)

    fields["indices"] = torchtext.data.Field(
        use_vocab=False, dtype=torch.long,
        sequential=False, tokenize=str.split)

    if dynamic_dict:
        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src_map, sequential=False, tokenize=str.split)

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False, tokenize=str.split)

    return fields


def vocab_to_fields(vocab, data_type="text"):
    """
    vocab: a list of (str, torchtext.vocab.Vocab) tuples (always?)
    data_type: text, img, or audio
    returns: a dict whose keys are the strings from the input vocab and whose
        values are fields created by a call to get_fields.
    """
    # this function is used:
    # 1) in model_builder.py, in the function load_test_model, which is used
    # in translator.py.
    # 2) in extract_embeddings.py, in order to build the model whose embeddings
    # will get extracted (this through a call to
    # model_builder.build_base_model, which needs the fields as an argument
    # to inputters.collect_feature_vocabs, which it uses so that it can
    # have access to the feature vocabularies.
    n_src_features = len([name for name, v in vocab if 'src_feat' in name])
    n_tgt_features = len([name for name, v in vocab if 'tgt_feat' in name])
    fields = get_fields(data_type, n_src_features, n_tgt_features)

    for name, v in vocab:
        # Hack. Can't pickle defaultdict :(
        # bp: this isn't true, you can pickle a defaultdict if the callable
        # is defined at the top level of the module. You can't pickle lambdas.
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[name].vocab = v
    return fields


def fields_to_vocab(fields):
    """
    fields: a dict whose keys are strings and whose values are Field objects
    returns: a list of 2-tuples whose first items are keys of the fields dict
             and whose values are the vocabs of the corresponding Fields.
    """
    # this is used in model_saver.py and preprocess.py
    return [(k, f.vocab) for k, f in fields.items()
            if f is not None and 'vocab' in f.__dict__]


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
    # this is used only in build_vocabs (and in the testing scripts)
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return Vocab(
        merged, max_size=vocab_size,
        specials=[UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD])


def num_features(corpus_file):
    """
    corpus_file (str): file path to get the features.
    returns the number of features for which to make fields
    """
    # used only in preprocess.py
    # in the long run tokenization (including for the features) should be
    # a field issue.
    with io.open(corpus_file, "r", encoding="utf-8") as f:
        feature_line = f.readline().strip().split()
        _, _, num_feats = extract_text_features(feature_line)
    return num_feats


def make_features(batch, side, data_type='text'):
    """
    Args:
        batch (Tensor): a batch of source or target data.
        side (str): for source or for target.
        data_type (str): type of the source input.
            Options are [text|img|audio].
    Returns:
        A sequence of src/tgt tensors with optional feature tensors
        of size (len x batch).
    """
    # used in trainer.py and translator.py
    # this should be a field issue
    assert side in ['src', 'tgt']
    if isinstance(batch.__dict__[side], tuple):
        data = batch.__dict__[side][0]
    else:
        data = batch.__dict__[side]

    feat_start = side + "_feat_"
    keys = sorted([k for k in batch.__dict__ if feat_start in k])
    features = [batch.__dict__[k] for k in keys]
    levels = [data] + features

    if data_type == 'text':
        return torch.cat([level.unsqueeze(2) for level in levels], 2)
    else:
        return levels[0]


def collect_feature_vocabs(fields, side):
    """
    fields: dict from strings to Field objects
    side: src or tgt
    returns: a list containing the Vocab objects belonging to feature fields
        on the given side
    """
    # used in train_single.py so that the model can build correct-sized
    # embedding matrices and in model_builder.py in load_test_model (for the
    # same reason, but at translation time)
    assert side in ['src', 'tgt']
    feature_vocabs = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feature_vocabs.append(fields[key].vocab)
    return feature_vocabs


def build_dataset(fields, data_type,
                  src_path=None, tgt_path=None,
                  src_lines=None, tgt_lines=None,
                  src_dir=None, src_seq_length=0, tgt_seq_length=0,
                  src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                  dynamic_dict=True, sample_rate=0,
                  window_size=0, window_stride=0, window=None,
                  normalize_audio=True, use_filter_pred=True):
    """
    A quasi-constructor for a Dataset. Give it a pair of paths and some kwargs,
    it'll make a Dataset object. Would it be better if this were a @classmethod
    alternate constructor for a Dataset class? Probably, but this is about
    incremental progress.
    """
    # used in preprocess.py and translator.py
    # in translator.py, path and iter arguments are passed to
    # build_dataset, but when it's actually called in translate.py the iter
    # arguments are also unspecified and thus None.

    src_data_classes = {'text': TextDataset, 'img': ImageDataset,
                        'audio': AudioDataset}
    assert data_type in src_data_classes

    assert src_path is not None or src_lines is not None
    assert src_path is None or src_lines is None

    assert tgt_path is None or tgt_lines is None

    # you can have a None tgt_path because you can translate without a gold tgt
    # make_examples is used only in this function
    if tgt_path or tgt_lines:
        tgt_examples = TextDataset.make_examples(
                path=tgt_path, lines=tgt_lines,
                truncate=tgt_seq_length_trunc, side="tgt")
    else:
        tgt_examples = None

    src_data_cls = src_data_classes[data_type]
    src_examples = src_data_cls.make_examples(
        path=src_path,
        lines=src_lines,
        truncate=src_seq_length_trunc,
        side="src", directory=src_dir,
        sample_rate=sample_rate, window_size=window_size,
        window_stride=window_stride, window=window,
        normalize_audio=normalize_audio)

    if use_filter_pred:
        # hack: better way is to check whether the src field is sequential
        use_src_len = data_type == 'text'
        fp = partial(
            filter_pred, use_src_len=use_src_len, use_tgt_len=True,
            min_src_len=0, max_src_len=src_seq_length,
            min_tgt_len=0, max_tgt_len=tgt_seq_length)
    else:
        fp = None

    dataset = src_data_cls(
        fields, src_examples, tgt_examples,
        dynamic_dict=dynamic_dict, filter_pred=fp)

    return dataset


def build_vocabs(datasets, data_type, share_vocab,
                 src_vocab_path, src_vocab_size, src_words_min_frequency,
                 tgt_vocab_path, tgt_vocab_size, tgt_words_min_frequency):
    """
    Args:
        datasets: a list of train dataset objects.
        fields (dict): dict whose keys are strings and values are Fields
        data_type: "text", "img" or "audio"?
        share_vocab(bool): share source and target vocabulary?
        src_vocab_path(string): Path to src vocabulary file.
        src_vocab_size(int): size of the source vocabulary.
        src_words_min_frequency(int): the minimum frequency needed to
                include a source word in the vocabulary.
        tgt_vocab_path(string): Path to tgt vocabulary file.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.

    Returns:
        fields, but after .build_vocab has been called for each field
    """
    # used only in preprocess.py

    src_vocab = _load_vocabulary(src_vocab_path, tag="source")
    tgt_vocab = _load_vocabulary(tgt_vocab_path, tag="target")

    fields = datasets[0].fields
    for name, field in fields.items():
        if field.use_vocab:
            if name == 'src':
                field.build_vocab(
                    *datasets, max_size=src_vocab_size,
                    min_freq=src_words_min_frequency)
                if src_vocab is not None:
                    field.vocab = _filtered_vocab(field.vocab, src_vocab)
            elif name == 'tgt':
                field.build_vocab(
                    *datasets, max_size=tgt_vocab_size,
                    min_freq=tgt_words_min_frequency)
                if tgt_vocab is not None:
                    field.vocab = _filtered_vocab(field.vocab, tgt_vocab)
            else:
                field.build_vocab(*datasets)

    if data_type == 'text':
        logger.info(" * src vocab size: %d." % len(fields["src"].vocab))
    logger.info(" * tgt vocab size: %d." % len(fields["tgt"].vocab))

    src_feats = sorted(name for name in fields if "src_feat_" in name)
    tgt_feats = sorted(name for name in fields if "tgt_feat_" in name)
    for i, feat in enumerate(src_feats, 1):
        logger.info(" * %s vocab size: %d." % (feat, len(fields[feat].vocab)))
    for i, feat in enumerate(tgt_feats, 1):
        logger.info(" * %s vocab size: %d." % (feat, len(fields[feat].vocab)))

    if data_type == 'text' and share_vocab:
        # Merge the input and output vocabularies.
        # `tgt_vocab_size` is ignored when sharing vocabularies
        logger.info(" * merging src and tgt vocab...")
        merged_vocab = merge_vocabs(
            [fields["src"].vocab, fields["tgt"].vocab],
            vocab_size=src_vocab_size)
        fields["src"].vocab = merged_vocab
        fields["tgt"].vocab = merged_vocab

    return fields


def _filtered_vocab(vocab, wordset):
    # used only in build_vocabs above when src or tgt vocab is predetermined
    counts = Counter({k: v for k, v in vocab.freqs.items() if k in wordset})
    return Vocab(counts, specials=[UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD])


def _load_vocabulary(vocab_path, tag=""):
    """
    Loads a vocabulary from the given path.
    :param vocabulary_path: path to load vocabulary from
    :param tag: tag for vocabulary (only used for logging)
    :return: vocabulary or None if path is null
    """
    if not vocab_path:
        return None
    if not os.path.exists(vocab_path):
        raise RuntimeError(
            "{} vocabulary not found at {}!".format(tag, vocab_path))
    logger.info("Loading {} vocabulary from {}".format(tag, vocab_path))
    with open(vocab_path) as f:
        return {line.strip().split()[0] for line in f if len(line.strip()) > 0}


class OrderedIterator(torchtext.data.Iterator):

    def create_batches(self):
        """ Create batches """
        if self.train:
            def _pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = _pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, batch_size, batch_size_fn, device, is_train):
        self.datasets = datasets
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # length of only the current dataset because this is lazy
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset

    def _next_dataset_iterator(self, dataset_iter):
        try:
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return OrderedIterator(
            dataset=self.cur_dataset, batch_size=self.batch_size,
            batch_size_fn=self.batch_size_fn,
            device=self.device, train=self.is_train,
            sort=False, sort_within_batch=True,
            repeat=False)


def build_dataset_iter(datasets, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over. We implement simple ordered iterator strategy here,
    but more sophisticated strategy like curriculum learning is ok too.
    """
    # used only in train_single.py in the return of the train_iter_fct
    # and valid_iter_fct functions, which uses lazily_load_dataset() as
    # the first argument.
    # In no case in the current code is is_train set to False
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    if is_train and opt.batch_type == "tokens":
        def batch_size_fn(new, count, sofar):
            """
            In token batching scheme, the number of sequences is limited
            such that the total number of src/tgt tokens (including padding)
            in a batch <= batch_size
            """
            # Maintains the longest src and tgt length in the current batch
            global max_src_in_batch, max_tgt_in_batch
            # Reset current longest length at a new batch (count=1)
            if count == 1:
                max_src_in_batch = 0
                max_tgt_in_batch = 0
            # Src: <bos> w1 ... wN <eos>
            max_src_in_batch = max(max_src_in_batch, len(new.src) + 2)
            # Tgt: w1 ... wN <eos>
            max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 1)
            src_elements = count * max_src_in_batch
            tgt_elements = count * max_tgt_in_batch
            return max(src_elements, tgt_elements)
    else:
        batch_size_fn = None
    device = "cuda" if opt.gpuid else "cpu"

    return DatasetLazyIter(datasets, batch_size, batch_size_fn,
                           device, is_train)


def lazily_load_dataset(corpus_type, path):
    """
    Args:
        corpus_type: 'train' or 'valid'
        path: the path that all serialized dataset .pt files match
    yields dataset objects
    """
    assert corpus_type in ["train", "valid"]

    # This is a lexicographic sort: data.train.11.pt is before data.train.2.pt
    for pt in sorted(glob.glob(path + '.' + corpus_type + '*.pt')):
        dataset = torch.load(pt)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt, len(dataset)))
        yield dataset


def shard_corpus(src_corpus, tgt_corpus, shard_size):
    """
    src_corpus: location of the corpus file which will be broken by shard_size
    tgt_corpus: location of the corpus file which will be broken into shards
        that are the same number of lines as the shards
    shard_size: size of each shard in bytes
    yields pairs of lists of lines where the first is approximately shard_size
        bytes and the second has the same number of lines as the first
    """
    assert shard_size != 0
    with io.open(src_corpus, "r", encoding="utf-8") as src_f, \
            io.open(tgt_corpus, "r", encoding="utf-8") as tgt_f:
        while True:
            src_shard = src_f.readlines(shard_size)
            if not src_shard:
                return
            tgt_shard = [next(tgt_f) for line in src_shard]
            yield src_shard, tgt_shard
