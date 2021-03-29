# -*- coding: utf-8 -*-
import os
import codecs
import math

from collections import Counter, defaultdict, OrderedDict

import torch
from torchtext.data import Field, RawField, LabelField
from torchtext.vocab import Vocab

from onmt.constants import DefaultTokens, ModelTask
from onmt.inputters.text_dataset import text_fields
from onmt.utils.logging import logger
# backwards compatibility
from onmt.inputters.text_dataset import _feature_tokenize  # noqa: F401

import gc


# monkey-patch to make torchtext Vocab's pickleable
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


class AlignField(LabelField):
    """
    Parse ['<src>-<tgt>', ...] into ['<src>','<tgt>', ...]
    """

    def __init__(self, **kwargs):
        kwargs['use_vocab'] = False
        kwargs['preprocessing'] = parse_align_idx
        super(AlignField, self).__init__(**kwargs)

    def process(self, batch, device=None):
        """ Turn a batch of align-idx to a sparse align idx Tensor"""
        sparse_idx = []
        for i, example in enumerate(batch):
            for src, tgt in example:
                # +1 for tgt side to keep coherent after "bos" padding,
                # register ['N°_in_batch', 'tgt_id+1', 'src_id']
                sparse_idx.append([i, tgt + 1, src])

        align_idx = torch.tensor(sparse_idx, dtype=self.dtype, device=device)

        return align_idx


def parse_align_idx(align_pharaoh):
    """
    Parse Pharaoh alignment into [[<src>, <tgt>], ...]
    """
    align_list = align_pharaoh.strip().split(' ')
    flatten_align_idx = []
    for align in align_list:
        try:
            src_idx, tgt_idx = align.split('-')
        except ValueError:
            logger.warning("{} in `{}`".format(align, align_pharaoh))
            logger.warning("Bad alignement line exists. Please check file!")
            raise
        flatten_align_idx.append([int(src_idx), int(tgt_idx)])
    return flatten_align_idx


def get_task_spec_tokens(data_task, pad, bos, eos):
    """
    Retrieve pad/bos/eos tokens for each data tasks
    """
    if data_task == ModelTask.SEQ2SEQ:
        return {
            "src": {"pad": pad, "bos": None, "eos": None},
            "tgt": {"pad": pad, "bos": bos, "eos": eos},
        }
    elif data_task == ModelTask.LANGUAGE_MODEL:
        return {
            "src": {"pad": pad, "bos": bos, "eos": None},
            "tgt": {"pad": pad, "bos": None, "eos": eos},
        }
    else:
        raise ValueError(f"No task specific tokens defined for {data_task}")


def get_fields(
    src_data_type,
    n_src_feats,
    n_tgt_feats,
    pad=DefaultTokens.PAD,
    bos=DefaultTokens.BOS,
    eos=DefaultTokens.EOS,
    dynamic_dict=False,
    with_align=False,
    src_truncate=None,
    tgt_truncate=None,
    data_task=ModelTask.SEQ2SEQ
):
    """
    Args:
        src_data_type: type of the source input. Options are [text].
        n_src_feats (int): the number of source features (not counting tokens)
            to create a :class:`torchtext.data.Field` for. (If
            ``src_data_type=="text"``, these fields are stored together
            as a ``TextMultiField``).
        n_tgt_feats (int): See above.
        pad (str): Special pad symbol. Used on src and tgt side.
        bos (str): Special beginning of sequence symbol. Only relevant
            for tgt.
        eos (str): Special end of sequence symbol. Only relevant
            for tgt.
        dynamic_dict (bool): Whether or not to include source map and
            alignment fields.
        with_align (bool): Whether or not to include word align.
        src_truncate: Cut off src sequences beyond this (passed to
            ``src_data_type``'s data reader - see there for more details).
        tgt_truncate: Cut off tgt sequences beyond this (passed to
            :class:`TextDataReader` - see there for more details).

    Returns:
        A dict mapping names to fields. These names need to match
        the dataset example attributes.
    """

    assert src_data_type in ['text'], \
        "Data type not implemented"
    assert not dynamic_dict or src_data_type == 'text', \
        'it is not possible to use dynamic_dict with non-text input'
    fields = {}

    fields_getters = {"text": text_fields}
    task_spec_tokens = get_task_spec_tokens(data_task, pad, bos, eos)

    src_field_kwargs = {
        "n_feats": n_src_feats,
        "include_lengths": True,
        "pad": task_spec_tokens["src"]["pad"],
        "bos": task_spec_tokens["src"]["bos"],
        "eos": task_spec_tokens["src"]["eos"],
        "truncate": src_truncate,
        "base_name": "src",
    }
    fields["src"] = fields_getters[src_data_type](**src_field_kwargs)

    tgt_field_kwargs = {
        "n_feats": n_tgt_feats,
        "include_lengths": False,
        "pad": task_spec_tokens["tgt"]["pad"],
        "bos": task_spec_tokens["tgt"]["bos"],
        "eos": task_spec_tokens["tgt"]["eos"],
        "truncate": tgt_truncate,
        "base_name": "tgt",
    }
    fields["tgt"] = fields_getters["text"](**tgt_field_kwargs)

    indices = Field(use_vocab=False, dtype=torch.long, sequential=False)
    fields["indices"] = indices

    if dynamic_dict:
        src_map = Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)
        fields["src_map"] = src_map

        src_ex_vocab = RawField()
        fields["src_ex_vocab"] = src_ex_vocab

        align = Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)
        fields["alignment"] = align

    if with_align:
        word_align = AlignField()
        fields["align"] = word_align

    return fields


class IterOnDevice(object):
    """Sent items from `iterable` on `device_id` and yield."""

    def __init__(self, iterable, device_id):
        self.iterable = iterable
        self.device_id = device_id

    @staticmethod
    def batch_to_device(batch, device_id):
        """Move `batch` to `device_id`, cpu if `device_id` < 0."""
        curr_device = batch.indices.device
        device = torch.device(device_id) if device_id >= 0 \
            else torch.device('cpu')
        if curr_device != device:
            if isinstance(batch.src, tuple):
                batch.src = tuple([_.to(device) for _ in batch.src])
            else:
                batch.src = batch.src.to(device)
            batch.tgt = batch.tgt.to(device)
            batch.indices = batch.indices.to(device)
            batch.alignment = batch.alignment.to(device) \
                if hasattr(batch, 'alignment') else None
            batch.src_map = batch.src_map.to(device) \
                if hasattr(batch, 'src_map') else None
            batch.align = batch.align.to(device) \
                if hasattr(batch, 'align') else None

    def __iter__(self):
        for batch in self.iterable:
            self.batch_to_device(batch, self.device_id)
            yield batch


def filter_example(ex, use_src_len=True, use_tgt_len=True,
                   min_src_len=1, max_src_len=float('inf'),
                   min_tgt_len=1, max_tgt_len=float('inf')):
    """Return whether an example is an acceptable length.

    If used with a dataset as ``filter_pred``, use :func:`partial()`
    for all keyword arguments.

    Args:
        ex (torchtext.data.Example): An object with a ``src`` and ``tgt``
            property.
        use_src_len (bool): Filter based on the length of ``ex.src``.
        use_tgt_len (bool): Similar to above.
        min_src_len (int): A non-negative minimally acceptable length
            (examples of exactly this length will be included).
        min_tgt_len (int): Similar to above.
        max_src_len (int or float): A non-negative (possibly infinite)
            maximally acceptable length (examples of exactly this length
            will be included).
        max_tgt_len (int or float): Similar to above.
    """

    src_len = len(ex.src[0])
    tgt_len = len(ex.tgt[0])
    return (not use_src_len or min_src_len <= src_len <= max_src_len) and \
        (not use_tgt_len or min_tgt_len <= tgt_len <= max_tgt_len)


def _pad_vocab_to_multiple(vocab, multiple):
    vocab_size = len(vocab)
    if vocab_size % multiple == 0:
        return
    target_size = int(math.ceil(vocab_size / multiple)) * multiple
    padding_tokens = ["{}{}".format(DefaultTokens.VOCAB_PAD, i)
                      for i in range(target_size - vocab_size)]
    vocab.extend(Vocab(Counter(), specials=padding_tokens))
    return vocab


def _build_field_vocab(field, counter, size_multiple=1, **kwargs):
    # this is basically copy-pasted from torchtext.
    all_special = [
        field.unk_token, field.pad_token, field.init_token, field.eos_token
    ]
    all_special.extend(list(kwargs.pop('specials', [])))
    specials = list(OrderedDict.fromkeys(
        tok for tok in all_special if tok is not None))
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)
    if size_multiple > 1:
        _pad_vocab_to_multiple(field.vocab, size_multiple)


def _load_vocab(vocab_path, name, counters, min_freq=0):
    """Inplace update `counters`[`name`] with vocab in `vocab_path`.

    Each line of `vocab_path` have a token, possible with a count.
    If not with count, each token will be assigned one so that the order
    of counters[name] will be same with `vocab_path`, and the minimum count
    number to be `min_freq` which defaults 0.
    """
    # counters changes in place
    vocab, has_count = _read_vocab_file(vocab_path, name)
    vocab_size = len(vocab)
    logger.info('Loaded %s vocab has %d tokens.' % (name, vocab_size))
    if not has_count:
        for i, token in enumerate(vocab):
            # keep the order of tokens specified in the vocab file by
            # adding them to the counter with decreasing counting values
            counters[name][token] = vocab_size - i + min_freq
    else:
        for token, count in vocab:
            counters[name][token] = int(count)
    return vocab, vocab_size


def _build_fv_from_multifield(multifield, counters, build_fv_kwargs,
                              size_multiple=1):
    for name, field in multifield:
        _build_field_vocab(
            field,
            counters[name],
            size_multiple=size_multiple,
            **build_fv_kwargs[name])
        logger.info(" * %s vocab size: %d." % (name, len(field.vocab)))


def _build_fields_vocab(fields, counters, data_type, share_vocab,
                        vocab_size_multiple,
                        src_vocab_size, src_words_min_frequency,
                        tgt_vocab_size, tgt_words_min_frequency,
                        src_specials=None, tgt_specials=None):
    src_specials = list(src_specials) if src_specials is not None else []
    tgt_specials = list(tgt_specials) if tgt_specials is not None else []
    build_fv_kwargs = defaultdict(dict)
    build_fv_kwargs["src"] = dict(
        max_size=src_vocab_size, min_freq=src_words_min_frequency,
        specials=src_specials)
    build_fv_kwargs["tgt"] = dict(
        max_size=tgt_vocab_size, min_freq=tgt_words_min_frequency,
        specials=tgt_specials)
    tgt_multifield = fields["tgt"]
    _build_fv_from_multifield(
        tgt_multifield,
        counters,
        build_fv_kwargs,
        size_multiple=vocab_size_multiple if not share_vocab else 1)

    if data_type == 'text':
        src_multifield = fields["src"]
        _build_fv_from_multifield(
            src_multifield,
            counters,
            build_fv_kwargs,
            size_multiple=vocab_size_multiple if not share_vocab else 1)

        if share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            logger.info(" * merging src and tgt vocab...")
            src_field = src_multifield.base_field
            tgt_field = tgt_multifield.base_field
            _all_specials = [item for item in src_specials + tgt_specials]
            _merge_field_vocabs(
                src_field, tgt_field, vocab_size=src_vocab_size,
                min_freq=src_words_min_frequency,
                vocab_size_multiple=vocab_size_multiple,
                specials=_all_specials)
            logger.info(" * merged vocab size: %d." % len(src_field.vocab))

    return fields


def build_vocab(train_dataset_files, fields, data_type, share_vocab,
                src_vocab_path, src_vocab_size, src_words_min_frequency,
                tgt_vocab_path, tgt_vocab_size, tgt_words_min_frequency,
                vocab_size_multiple=1):
    """Build the fields for all data sides.

    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict[str, Field]): fields to build vocab for.
        data_type (str): A supported data type string.
        share_vocab (bool): share source and target vocabulary?
        src_vocab_path (str): Path to src vocabulary file.
        src_vocab_size (int): size of the source vocabulary.
        src_words_min_frequency (int): the minimum frequency needed to
            include a source word in the vocabulary.
        tgt_vocab_path (str): Path to tgt vocabulary file.
        tgt_vocab_size (int): size of the target vocabulary.
        tgt_words_min_frequency (int): the minimum frequency needed to
            include a target word in the vocabulary.
        vocab_size_multiple (int): ensure that the vocabulary size is a
            multiple of this value.

    Returns:
        Dict of Fields
    """

    counters = defaultdict(Counter)

    if src_vocab_path:
        try:
            logger.info("Using existing vocabulary...")
            vocab = torch.load(src_vocab_path)
            # return vocab to dump with standard name
            return vocab
        except torch.serialization.pickle.UnpicklingError:
            logger.info("Building vocab from text file...")
            # empty train_dataset_files so that vocab is only loaded from
            # given paths in src_vocab_path, tgt_vocab_path
            train_dataset_files = []

    # Load vocabulary
    if src_vocab_path:
        src_vocab, src_vocab_size = _load_vocab(
            src_vocab_path, "src", counters,
            src_words_min_frequency)
    else:
        src_vocab = None

    if tgt_vocab_path:
        tgt_vocab, tgt_vocab_size = _load_vocab(
            tgt_vocab_path, "tgt", counters,
            tgt_words_min_frequency)
    else:
        tgt_vocab = None

    for i, path in enumerate(train_dataset_files):
        dataset = torch.load(path)
        logger.info(" * reloading %s." % path)
        for ex in dataset.examples:
            for name, field in fields.items():
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

    fields = _build_fields_vocab(
        fields, counters, data_type,
        share_vocab, vocab_size_multiple,
        src_vocab_size, src_words_min_frequency,
        tgt_vocab_size, tgt_words_min_frequency)

    return fields  # is the return necessary?


def _merge_field_vocabs(src_field, tgt_field, vocab_size, min_freq,
                        vocab_size_multiple, specials):
    # in the long run, shouldn't it be possible to do this by calling
    # build_vocab with both the src and tgt data?
    init_specials = [tgt_field.unk_token, tgt_field.pad_token,
                     tgt_field.init_token, tgt_field.eos_token]
    all_specials = list(OrderedDict.fromkeys(
        tok for tok in init_specials + specials
        if tok is not None))
    merged = sum(
        [src_field.vocab.freqs, tgt_field.vocab.freqs], Counter()
    )
    merged_vocab = Vocab(
        merged, specials=all_specials,
        max_size=vocab_size, min_freq=min_freq
    )
    if vocab_size_multiple > 1:
        _pad_vocab_to_multiple(merged_vocab, vocab_size_multiple)
    src_field.vocab = merged_vocab
    tgt_field.vocab = merged_vocab
    assert len(src_field.vocab) == len(tgt_field.vocab)


def _read_vocab_file(vocab_path, tag):
    """Loads a vocabulary from the given path.

    Args:
        vocab_path (str): Path to utf-8 text file containing vocabulary.
            Each token should be on a line, may followed with a count number
            seperate by space if `with_count`. No extra whitespace is allowed.
        tag (str): Used for logging which vocab is being read.
    """

    logger.info("Loading {} vocabulary from {}".format(tag, vocab_path))

    if not os.path.exists(vocab_path):
        raise RuntimeError(
            "{} vocabulary not found at {}".format(tag, vocab_path))
    else:
        with codecs.open(vocab_path, 'r', 'utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            first_line = lines[0].split(None, 1)
            has_count = (len(first_line) == 2 and first_line[-1].isdigit())
            if has_count:
                vocab = [line.split(None, 1) for line in lines]
            else:
                vocab = [line.strip().split()[0] for line in lines]
            return vocab, has_count
