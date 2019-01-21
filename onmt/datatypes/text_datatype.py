# -*- coding: utf-8 -*-
from functools import partial

import torch
from torchtext.data import Field

from onmt.datatypes.datareader_base import DataReaderBase
from onmt.datatypes.datatype_base import Datatype


class TextDataReader(DataReaderBase):
    """
    Build `Example` objects, `Field` objects, and filter_pred function
    from text corpus.
    """

    def read(self, sequences, side, src_dir=None):
        """Read lines of text.

        Args:
            sequences: path to corpus file or iterable
            side (str): "src" or "tgt".

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """

        assert src_dir is None or src_dir == '', \
            "Cannot use src_dir with text dataset. (Got " \
            "{:s}, but expected None or '')".format(src_dir.__str__())

        if isinstance(sequences, str):
            sequences = self._read_file(sequences)
        for i, seq in enumerate(sequences):
            yield {side: seq, "indices": i}


def text_sort_key(ex):
    if hasattr(ex, "tgt"):
        return len(ex.src), len(ex.tgt)
    return len(ex.src)


# mix this with partial
def _feature_tokenize(
        string, layer=0, tok_delim=None, feat_delim=None, truncate=None):
    tokens = string.split(tok_delim)
    if truncate is not None:
        tokens = tokens[:truncate]
    if feat_delim is not None:
        tokens = [t.split(feat_delim)[layer] for t in tokens]
    return tokens


def text_fields(base_name, **kwargs):
    """Create text fields.

    Args:
        base_name (str)
        n_feats (int)
        include_lengths (bool)
        pad (str, optional): Defaults to <blank>.
        bos (str or NoneType, optional): Defaults to <s>
        eos (str or NoneType, optional): Defaults to </s>
        truncate (bool or NoneType, optional): Defaults to None.
    """

    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    pad = kwargs.get("pad", "<blank>")
    bos = kwargs.get("bos", "<s>")
    eos = kwargs.get("eos", "</s>")
    truncate = kwargs.get("truncate", None)
    fields_ = []
    feat_delim = u"￨" if n_feats > 0 else None
    for i in range(n_feats + 1):
        name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name
        tokenize = partial(
            _feature_tokenize,
            layer=i,
            truncate=truncate,
            feat_delim=feat_delim)
        use_len = i == 0 and include_lengths
        feat = Field(
            init_token=bos, eos_token=eos,
            pad_token=pad, tokenize=tokenize,
            include_lengths=use_len)
        fields_.append((name, feat))
    return [], fields_


text = Datatype("text", TextDataReader, text_sort_key, text_fields)
