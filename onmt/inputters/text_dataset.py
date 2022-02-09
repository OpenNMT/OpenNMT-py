# -*- coding: utf-8 -*-
from functools import partial
from itertools import repeat

import torch
from torchtext.data import Field, RawField

from onmt.constants import DefaultTokens
from onmt.inputters.datareader_base import DataReaderBase
from onmt.utils.misc import split_corpus


class TextDataReader(DataReaderBase):
    def read(self, sequences, side, features={}):
        """Read text data from disk.
            Args:
            sequences (str or Iterable[str]):
                path to text file or iterable of the actual text data.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            features: (Dict[str or Iterable[str]]):
                dictionary mapping feature names with the path to feature
                file or iterable of the actual feature data.
        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)

        features_names = []
        features_values = []
        for feat_name, v in features.items():
            features_names.append(feat_name)
            if isinstance(v, str):
                features_values.append(DataReaderBase._read_file(features))
            else:
                features_values.append(v)
        for i, (seq, *feats) in enumerate(zip(sequences, *features_values)):
            ex_dict = {}
            if isinstance(seq, bytes):
                seq = seq.decode("utf-8")
            ex_dict[side] = seq
            for j, f in enumerate(feats):
                if isinstance(f, bytes):
                    f = f.decode("utf-8")
                ex_dict[features_names[j]] = f
            yield {side: ex_dict, "indices": i}


class InferenceDataReader(object):
    """It handles inference data reading from disk in shards.

    Args:
        src (str): path to the source file
        tgt (str or NoneType): path to the target file
        src_feats (Dict[str]): paths to the features files
        shard_size (int): divides files into smaller files of size shard_size

    Returns:
        Tuple[List[str], List[str], Dict[List[str]]]
    """

    def __init__(self, src, tgt, src_feats={}, shard_size=10000):
        self.src = src
        self.tgt = tgt
        self.src_feats = src_feats
        self.shard_size = shard_size

    def __iter__(self):
        src_shards = split_corpus(self.src, self.shard_size)
        tgt_shards = split_corpus(self.tgt, self.shard_size)

        if not self.src_feats:
            features_shards = [repeat(None)]
        else:
            features_shards = []
            features_names = []
            for feat_name, feat_path in self.src_feats.items():
                features_shards.append(
                    split_corpus(feat_path, self.shard_size))
                features_names.append(feat_name)

        shard_pairs = zip(src_shards, tgt_shards, *features_shards)
        for i, shard in enumerate(shard_pairs):
            src_shard, tgt_shard, *features_shard = shard
            if features_shard[0] is not None:
                features_shard_ = dict()
                for j, x in enumerate(features_shard):
                    features_shard_[features_names[j]] = x
            else:
                features_shard_ = None
            yield src_shard, tgt_shard, features_shard_


class InferenceDataIterator(object):

    def __init__(self, src, tgt, src_feats, transform):
        self.src = src
        self.tgt = tgt
        self.src_feats = src_feats
        self.transform = transform

    def _tokenize(self, example):
        example['src'] = example['src'].decode("utf-8").strip('\n').split()
        example['tgt'] = example['tgt'].decode("utf-8").strip('\n').split() \
            if example["tgt"] is not None else None
        example['src_original'] = example['src']
        example['tgt_original'] = example['tgt']
        if 'src_feats' in example:
            for k in example['src_feats'].keys():
                example['src_feats'][k] = example['src_feats'][k] \
                    .decode("utf-8").strip('\n').split() \
                    if example['src_feats'][k] is not None else None
        return example

    def _transform(self, example, remove_tgt=False):
        maybe_example = self.transform.apply(
                example, is_train=False, corpus_name="translate")
        assert maybe_example is not None, \
            "Transformation on example skipped the example. " \
            "Please check the transforms."
        return maybe_example

    def _process(self, example, remove_tgt=False):
        example['src'] = {"src": ' '.join(example['src'])}
        example['tgt'] = {"tgt": ' '.join(example['tgt'])}

        # Make features part of src as in TextMultiField
        # {'src': {'src': ..., 'feat1': ...., 'feat2': ....}}
        if 'src_feats' in example:
            for feat_name, feat_value in example['src_feats'].items():
                example['src'][feat_name] = ' '.join(feat_value)
            del example["src_feats"]

        # Cleanup
        if remove_tgt:
            del example["tgt"]
        del example["tgt_original"]
        del example["src_original"]

        return example

    def __iter__(self):
        tgt = self.tgt if self.tgt is not None else repeat(None)

        if self.src_feats is not None:
            features_names = []
            features_values = []
            for feat_name, values in self.src_feats.items():
                features_names.append(feat_name)
                features_values.append(values)
        else:
            features_values = [repeat(None)]

        for i, (src, tgt, *src_feats) in enumerate(zip(
                self.src, tgt, *features_values)):
            ex = {
                "src": src,
                "tgt": tgt if tgt is not None else b""
            }
            if src_feats[0] is not None:
                src_feats_ = {}
                for j, x in enumerate(src_feats):
                    src_feats_[features_names[j]] = x
                ex["src_feats"] = src_feats_
            ex = self._tokenize(ex)
            ex = self._transform(ex)
            ex = self._process(ex, remove_tgt=self.tgt is None)
            ex["indices"] = i
            yield ex


def text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if hasattr(ex, "tgt"):
        return len(ex.src[0]), len(ex.tgt[0])
    return len(ex.src[0])


# Legacy function. Currently it only truncates input if truncate is set.
# mix this with partial
def _feature_tokenize(
        string, layer=0, tok_delim=None, feat_delim=None, truncate=None):
    """Split apart word features (like POS/NER tags) from the tokens.

    Args:
        string (str): A string with ``tok_delim`` joining tokens and
            features joined by ``feat_delim``. For example,
            ``"hello|NOUN|'' Earth|NOUN|PLANET"``.
        layer (int): Which feature to extract. (Not used if there are no
            features, indicated by ``feat_delim is None``). In the
            example above, layer 2 is ``'' PLANET``.
        truncate (int or NoneType): Restrict sequences to this length of
            tokens.

    Returns:
        List[str] of tokens.
    """

    tokens = string.split(tok_delim)
    if truncate is not None:
        tokens = tokens[:truncate]
    if feat_delim is not None:
        tokens = [t.split(feat_delim)[layer] for t in tokens]
    return tokens


class TextMultiField(RawField):
    """Container for subfields.

    Text data might use POS/NER/etc labels in addition to tokens.
    This class associates the "base" :class:`Field` with any subfields.
    It also handles padding the data and stacking it.

    Args:
        base_name (str): Name for the base field.
        base_field (Field): The token field.
        feats_fields (Iterable[Tuple[str, Field]]): A list of name-field
            pairs.

    Attributes:
        fields (Iterable[Tuple[str, Field]]): A list of name-field pairs.
            The order is defined as the base field first, then
            ``feats_fields`` in alphabetical order.
    """

    def __init__(self, base_name, base_field, feats_fields):
        super(TextMultiField, self).__init__()
        self.fields = [(base_name, base_field)]
        for name, ff in sorted(feats_fields, key=lambda kv: kv[0]):
            self.fields.append((name, ff))

    @property
    def base_field(self):
        return self.fields[0][1]

    def process(self, batch, device=None):
        """Convert outputs of preprocess into Tensors.

        Args:
            batch (List[List[List[str]]]): A list of length batch size.
                Each element is a list of the preprocess results for each
                field (which are lists of str "words" or feature tags.
            device (torch.device or str): The device on which the tensor(s)
                are built.

        Returns:
            torch.LongTensor or Tuple[LongTensor, LongTensor]:
                A tensor of shape ``(seq_len, batch_size, len(self.fields))``
                where the field features are ordered like ``self.fields``.
                If the base field returns lengths, these are also returned
                and have shape ``(batch_size,)``.
        """

        # batch (list(list(list))): batch_size x len(self.fields) x seq_len
        batch_by_feat = list(zip(*batch))
        base_data = self.base_field.process(batch_by_feat[0], device=device)
        if self.base_field.include_lengths:
            # lengths: batch_size
            base_data, lengths = base_data

        feats = [ff.process(batch_by_feat[i], device=device)
                 for i, (_, ff) in enumerate(self.fields[1:], 1)]
        levels = [base_data] + feats
        # data: seq_len x batch_size x len(self.fields)
        data = torch.stack(levels, 2)
        if self.base_field.include_lengths:
            return data, lengths
        else:
            return data

    def preprocess(self, x):
        """Preprocess data.

        Args:
            x (str): A sentence string (words joined by whitespace).

        Returns:
            List[List[str]]: A list of length ``len(self.fields)`` containing
                lists of tokens/feature tags for the sentence. The output
                is ordered like ``self.fields``.
        """
        return [f.preprocess(x[fn]) for fn, f in self.fields]

    def __getitem__(self, item):
        return self.fields[item]


def text_fields(**kwargs):
    """Create text fields.

    Args:
        base_name (str): Name associated with the field.
        feats (Optional[Dict]): Word level feats
        include_lengths (bool): Optionally return the sequence lengths.
        pad (str, optional): Defaults to ``"<blank>"``.
        bos (str or NoneType, optional): Defaults to ``"<s>"``.
        eos (str or NoneType, optional): Defaults to ``"</s>"``.
        truncate (bool or NoneType, optional): Defaults to ``None``.

    Returns:
        TextMultiField
    """

    feats = kwargs["feats"]
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    pad = kwargs.get("pad", DefaultTokens.PAD)
    bos = kwargs.get("bos", DefaultTokens.BOS)
    eos = kwargs.get("eos", DefaultTokens.EOS)
    truncate = kwargs.get("truncate", None)
    fields_ = []

    feat_delim = None  # u"￨" if n_feats > 0 else None

    # Base field
    tokenize = partial(
        _feature_tokenize,
        layer=None,
        truncate=truncate,
        feat_delim=feat_delim)
    feat = Field(
        init_token=bos, eos_token=eos,
        pad_token=pad, tokenize=tokenize,
        include_lengths=include_lengths)
    fields_.append((base_name, feat))

    # Feats fields
    if feats:
        for feat_name in feats.keys():
            # Legacy function, it is not really necessary
            tokenize = partial(
                _feature_tokenize,
                layer=None,
                truncate=truncate,
                feat_delim=feat_delim)
            feat = Field(
                init_token=bos, eos_token=eos,
                pad_token=pad, tokenize=tokenize,
                include_lengths=False)
            fields_.append((feat_name, feat))

    assert fields_[0][0] == base_name  # sanity check
    field = TextMultiField(fields_[0][0], fields_[0][1], fields_[1:])
    return field
