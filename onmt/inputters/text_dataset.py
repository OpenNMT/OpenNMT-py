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
