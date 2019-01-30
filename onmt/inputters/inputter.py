# -*- coding: utf-8 -*-
import glob
import os
import codecs
import math

from collections import Counter, defaultdict
from itertools import chain, cycle
from functools import partial

import torch
import torchtext.data
from torchtext.data import Field
from torchtext.vocab import Vocab

from onmt.inputters.text_dataset import TextDataset, text_fields,\
    TextMultiField
from onmt.inputters.image_dataset import ImageDataset, image_fields
from onmt.inputters.audio_dataset import AudioDataset, audio_fields
from onmt.utils.logging import logger
# backwards compatibility
from onmt.inputters.text_dataset import _feature_tokenize  # noqa: F401
from onmt.inputters.image_dataset import (  # noqa: F401
    batch_img as make_img)

import gc


def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


Vocab.__getstate__ = _getstate
Vocab.__setstate__ = _setstate


def make_src(data, vocab):
    src_size = max([t.size(0) for t in data])
    src_vocab_size = max([t.max() for t in data]) + 1
    alignment = torch.zeros(src_size, len(data), src_vocab_size)
    for i, sent in enumerate(data):
        for j, t in enumerate(sent):
            alignment[j, i, t] = 1
    return alignment


def make_tgt(data, vocab):
    tgt_size = max([t.size(0) for t in data])
    alignment = torch.zeros(tgt_size, len(data)).long()
    for i, sent in enumerate(data):
        alignment[:sent.size(0), i] = sent
    return alignment


def get_fields(
    src_data_type,
    n_src_feats,
    n_tgt_feats,
    pad='<blank>',
    bos='<s>',
    eos='</s>',
    dynamic_dict=False,
    src_truncate=None,
    tgt_truncate=None
):
    """
    src_data_type: type of the source input. Options are [text|img|audio].
    n_src_feats, n_tgt_feats: the number of source and target features to
        create a `torchtext.data.Field` for.
    pad, bos, eos: special symbols to use for fields.
    returns: A dictionary. The keys are strings whose names correspond to the
        keys of the dictionaries yielded by the make_examples methods of
        various dataset classes. The values are lists of (name, Field)
        pairs, where the name is a string which will become the name of
        an attribute of an example.
    """
    assert src_data_type in ['text', 'img', 'audio'], \
        "Data type not implemented"
    assert not dynamic_dict or src_data_type == 'text', \
        'it is not possible to use dynamic_dict with non-text input'
    fields = {'src': [], 'tgt': []}

    fields_getters = {"text": text_fields,
                      "img": image_fields,
                      "audio": audio_fields}

    src_field_kwargs = {"n_feats": n_src_feats,
                        "include_lengths": True,
                        "pad": pad, "bos": None, "eos": None,
                        "truncate": src_truncate}
    fields["src"] = fields_getters[src_data_type](
        'src', **src_field_kwargs)

    tgt_field_kwargs = {"n_feats": n_tgt_feats,
                        "include_lengths": False,
                        "pad": pad, "bos": bos, "eos": eos,
                        "truncate": tgt_truncate}
    fields['tgt'] = fields_getters["text"](
        'tgt', **tgt_field_kwargs)

    indices = Field(use_vocab=False, dtype=torch.long, sequential=False)
    fields["indices"] = [('indices', indices)]

    if dynamic_dict:
        src_map = Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)
        fields["src_map"] = [("src_map", src_map)]

        align = Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)
        fields["alignment"] = [('alignment', align)]

    return fields


def load_old_vocab(vocab, data_type="text", dynamic_dict=False):
    """
    vocab: a list of (field name, torchtext.vocab.Vocab) pairs. This is the
           format formerly saved in *.vocab.pt files.
    data_type: text, img, or audio
    returns: a dictionary whose keys are the field names and whose values
             are lists of (name, Field) pairs
    """
    if _old_style_field_list(vocab):  # upgrade to multifield
        fields = vocab
        for base_name, vals in fields.items():
            if ((base_name == 'src' and data_type == 'text') or
                    base_name == 'tgt'):
                assert not isinstance(vals[0][1], TextMultiField)
                fields[base_name] = [(base_name, TextMultiField(
                    vals[0][0], vals[0][1], vals[1:]))]
        return fields
    vocab = dict(vocab)
    n_src_features = sum('src_feat_' in k for k in vocab)
    n_tgt_features = sum('tgt_feat_' in k for k in vocab)
    fields = get_fields(
        data_type, n_src_features, n_tgt_features, dynamic_dict=dynamic_dict
    )
    for k, vals in fields.items():
        for n, f in vals:
            try:
                f_iter = iter(f)
            except TypeError:
                f_iter = [(n, f)]
            for sub_n, sub_f in f_iter:
                if sub_n in vocab:
                    sub_f.vocab = vocab[sub_n]
    return fields


def _old_style_vocab(vocab):
    """
    vocab: some object loaded from a *.vocab.pt file
    returns: whether the object is a list of pairs where the second object
        is a torchtext.vocab.Vocab object.

    This exists because previously only the vocab objects from the fields
    were saved directly, not the fields themselves, and the fields needed to
    be reconstructed at training and translation time.
    """
    return isinstance(vocab, list) and \
        any(isinstance(v[1], Vocab) for v in vocab)


def _old_style_field_list(vocab):
    # if tgt isn't using TextMultiField, then no text field is.
    return not _old_style_vocab(vocab) and not isinstance(
        vocab['tgt'][0][1], TextMultiField)


def old_style_vocab(vocab):
    return _old_style_vocab(vocab) or _old_style_field_list(vocab)


def filter_example(ex, use_src_len=True, use_tgt_len=True,
                   min_src_len=1, max_src_len=float('inf'),
                   min_tgt_len=1, max_tgt_len=float('inf')):
    """
    A generalized function for filtering examples based on the length of their
    src or tgt values. Rather than being used by itself as the filter_pred
    argument to a dataset, it should be partially evaluated with everything
    specified except the value of the example.
    """
    src_len = len(ex.src[0])
    tgt_len = len(ex.tgt[0])
    return (not use_src_len or min_src_len <= src_len <= max_src_len) and \
        (not use_tgt_len or min_tgt_len <= tgt_len <= max_tgt_len)


def build_dataset(fields, data_type, src,
                  src_dir=None, tgt=None,
                  src_seq_len=50, tgt_seq_len=50,
                  sample_rate=0, window_size=0, window_stride=0, window=None,
                  normalize_audio=True, use_filter_pred=True,
                  image_channel_size=3):
    """
    src: path to corpus file or iterator over source data
    tgt: path to corpus file, iterator over target data, or None
    """
    dataset_classes = {
        'text': TextDataset, 'img': ImageDataset, 'audio': AudioDataset
    }
    assert data_type in dataset_classes
    assert src is not None
    if data_type == 'text':
        src_examples_iter = TextDataset.make_examples(src, "src")
    elif data_type == 'img':
        # there is a truncate argument as well, but it was never set to
        # anything besides None before
        src_examples_iter = ImageDataset.make_examples(
            src, src_dir, 'src', channel_size=image_channel_size
        )
    else:
        src_examples_iter = AudioDataset.make_examples(
            src, src_dir, "src", sample_rate,
            window_size, window_stride, window,
            normalize_audio, None)

    if tgt is None:
        tgt_examples_iter = None
    else:
        tgt_examples_iter = TextDataset.make_examples(tgt, "tgt")

    # the second conjunct means nothing will be filtered at translation time
    # if there is no target data
    if use_filter_pred and tgt_examples_iter is not None:
        filter_pred = partial(
            filter_example, use_src_len=data_type == 'text',
            max_src_len=src_seq_len, max_tgt_len=tgt_seq_len
        )
    else:
        filter_pred = None

    dataset_cls = dataset_classes[data_type]
    dataset = dataset_cls(
        fields, src_examples_iter, tgt_examples_iter, filter_pred=filter_pred)
    return dataset


def _pad_vocab_to_multiple(vocab, multiple):
    vocab_size = len(vocab)
    if vocab_size % multiple == 0:
        return
    target_size = int(math.ceil(vocab_size / multiple)) * multiple
    padding_tokens = [
        "averyunlikelytoken%d" % i for i in range(target_size - vocab_size)]
    vocab.extend(Vocab(Counter(), specials=padding_tokens))
    return vocab


def _build_field_vocab(field, counter, size_multiple=1, **kwargs):
    # this is basically copy-pasted from torchtext.
    all_specials = [
        field.unk_token, field.pad_token, field.init_token, field.eos_token
    ]
    specials = [tok for tok in all_specials if tok is not None]
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)
    if size_multiple > 1:
        _pad_vocab_to_multiple(field.vocab, size_multiple)


def _load_vocab(vocab_path, name, counters):
    # counters changes in place
    vocab = _read_vocab_file(vocab_path, name)
    vocab_size = len(vocab)
    logger.info('Loaded %s vocab has %d tokens.' % (name, vocab_size))
    for i, token in enumerate(vocab):
        # keep the order of tokens specified in the vocab file by
        # adding them to the counter with decreasing counting values
        counters[name][token] = vocab_size - i
    return vocab, vocab_size


def _build_fv_from_multifield(multifield, counters, build_fv_args,
                              size_multiple=1):
    for name, field in multifield:
        _build_field_vocab(
            field,
            counters[name],
            size_multiple=size_multiple,
            **build_fv_args[name])
        logger.info(" * %s vocab size: %d." % (name, len(field.vocab)))


def build_vocab(train_dataset_files, fields, data_type, share_vocab,
                src_vocab_path, src_vocab_size, src_words_min_frequency,
                tgt_vocab_path, tgt_vocab_size, tgt_words_min_frequency,
                vocab_size_multiple=1):
    """
    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict): fields to build vocab for.
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
        vocab_size_multiple(int): ensure that the vocabulary size is a multiple
                of this value.

    Returns:
        Dict of Fields
    """
    counters = defaultdict(Counter)

    # Load vocabulary
    if src_vocab_path:
        src_vocab, src_vocab_size = _load_vocab(
            src_vocab_path, "src", counters)
    else:
        src_vocab = None

    if tgt_vocab_path:
        tgt_vocab, tgt_vocab_size = _load_vocab(
            tgt_vocab_path, "tgt", counters)
    else:
        tgt_vocab = None

    for i, path in enumerate(train_dataset_files):
        dataset = torch.load(path)
        logger.info(" * reloading %s." % path)
        for ex in dataset.examples:
            for name, field in chain.from_iterable(fields.values()):
                try:
                    f_iter = iter(field)
                except TypeError:
                    f_iter = [(name, field)]
                    all_data = [getattr(ex, name, None)]
                else:
                    all_data = getattr(ex, name)
                for (sub_n, sub_f), fd in zip(
                        f_iter, all_data):
                    has_vocab = (sub_n == 'src' and src_vocab) or \
                                (sub_n == 'tgt' and tgt_vocab)
                    if sub_f.sequential and not has_vocab:
                        val = fd
                        counters[sub_n].update(val)

        # Drop the none-using from memory but keep the last
        if i < len(train_dataset_files) - 1:
            dataset.examples = None
            gc.collect()
            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

    build_fv_args = defaultdict(dict)
    build_fv_args["src"] = dict(
        max_size=src_vocab_size, min_freq=src_words_min_frequency)
    build_fv_args["tgt"] = dict(
        max_size=tgt_vocab_size, min_freq=tgt_words_min_frequency)
    assert len(fields["tgt"]) == 1
    tgt_multifield = fields["tgt"][0][1]
    _build_fv_from_multifield(
        tgt_multifield,
        counters,
        build_fv_args,
        size_multiple=vocab_size_multiple if not share_vocab else 1)
    if data_type == 'text':
        assert len(fields["src"]) == 1
        src_multifield = fields["src"][0][1]
        _build_fv_from_multifield(
            src_multifield,
            counters,
            build_fv_args,
            size_multiple=vocab_size_multiple if not share_vocab else 1)
        if share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            logger.info(" * merging src and tgt vocab...")
            src_field = src_multifield.base_field
            tgt_field = tgt_multifield.base_field
            _merge_field_vocabs(
                src_field, tgt_field, vocab_size=src_vocab_size,
                min_freq=src_words_min_frequency,
                vocab_size_multiple=vocab_size_multiple)
            logger.info(" * merged vocab size: %d." % len(src_field.vocab))
    return fields  # is the return necessary?


def _merge_field_vocabs(src_field, tgt_field, vocab_size, min_freq,
                        vocab_size_multiple):
    # in the long run, shouldn't it be possible to do this by calling
    # build_vocab with both the src and tgt data?
    specials = [tgt_field.unk_token, tgt_field.pad_token,
                tgt_field.init_token, tgt_field.eos_token]
    merged = sum(
        [src_field.vocab.freqs, tgt_field.vocab.freqs], Counter()
    )
    merged_vocab = Vocab(
        merged, specials=specials,
        max_size=vocab_size, min_freq=min_freq
    )
    if vocab_size_multiple > 1:
        _pad_vocab_to_multiple(merged_vocab, vocab_size_multiple)
    src_field.vocab = merged_vocab
    tgt_field.vocab = merged_vocab
    assert len(src_field.vocab) == len(tgt_field.vocab)


def _read_vocab_file(vocab_path, tag):
    """
    Loads a vocabulary from the given path.
    :param vocabulary_path: path to load vocabulary from
    :param tag: tag for vocabulary (only used for logging)
    :return: vocabulary or None if path is null
    """
    logger.info("Loading {} vocabulary from {}".format(tag, vocab_path))

    if not os.path.exists(vocab_path):
        raise RuntimeError(
            "{} vocabulary not found at {}".format(tag, vocab_path))
    else:
        with codecs.open(vocab_path, 'r', 'utf-8') as f:
            return [line.strip().split()[0] for line in f if line.strip()]


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
    """
    dataset_paths: a list containing the locations of datasets
    fields (dict): fields dict for the datasets.
    batch_size (int): batch size.
    batch_size_fn: custom batch process function.
    device: the GPU device.
    is_train (bool): train or valid?
    """

    def __init__(self, dataset_paths, fields, batch_size, batch_size_fn,
                 device, is_train):
        self._paths = dataset_paths
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

    def __iter__(self):
        paths = cycle(self._paths) if self.is_train else self._paths
        for path in paths:
            cur_dataset = torch.load(path)
            logger.info('Loading dataset from %s, number of examples: %d' %
                        (path, len(cur_dataset)))
            cur_dataset.fields = self.fields
            cur_iter = OrderedIterator(
                dataset=cur_dataset,
                batch_size=self.batch_size,
                batch_size_fn=self.batch_size_fn,
                device=self.device,
                train=self.is_train,
                sort=False,
                sort_within_batch=True,
                repeat=False
            )
            for batch in cur_iter:
                yield batch

            cur_dataset.examples = None
            gc.collect()
            del cur_dataset
            gc.collect()


def max_tok_len(new, count, sofar):
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
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wN <eos>]
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt[0]) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def build_dataset_iter(corpus_type, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over. We implement simple ordered iterator strategy here,
    but more sophisticated strategy like curriculum learning is ok too.
    """
    dataset_paths = sorted(glob.glob(opt.data + '.' + corpus_type + '*.pt'))
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_fn = max_tok_len if is_train and opt.batch_type == "tokens" else None

    device = "cuda" if opt.gpu_ranks else "cpu"

    return DatasetLazyIter(dataset_paths, fields, batch_size, batch_fn,
                           device, is_train)
