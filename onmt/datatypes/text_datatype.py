# -*- coding: utf-8 -*-

import torch

from onmt.datatypes.datareader_base import DataReaderBase


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

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs,
                             batch_dim=1, batch_offset=None):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambiguous.
        """
        offset = len(tgt_vocab)
        for b in range(scores.size(batch_dim)):
            blank = []
            fill = []
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            src_vocab = src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices.data)
                fill = torch.Tensor(fill).type_as(batch.indices.data)
                score = scores[:, b] if batch_dim == 1 else scores[b]
                score.index_add_(1, fill, score.index_select(1, blank))
                score.index_fill_(1, blank, 1e-10)
        return scores


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
