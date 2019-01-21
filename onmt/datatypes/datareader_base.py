# coding: utf-8
import codecs


class DataReaderBase(object):
    @classmethod
    def from_opt(cls, opt):
        return cls()

    def read(self, src, side, src_dir=None):
        raise NotImplementedError()

    @staticmethod
    def _read_file(path):
        with codecs.open(path, "r", "utf-8") as f:
            for line in f:
                yield line
