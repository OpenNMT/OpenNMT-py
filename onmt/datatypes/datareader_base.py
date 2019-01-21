# coding: utf-8
import codecs


# several data readers need optional dependencies. There's no
# appropriate standard exception
class MissingDependencyException(Exception):
    pass


class DataReaderBase(object):
    @classmethod
    def from_opt(cls, opt):
        return cls()

    @staticmethod
    def _raise_missing_dep(*missing_deps):
        """Raise missing dep exception with standard error message."""
        raise MissingDependencyException(
            "Could not create reader. Be sure to install "
            "the following dependencies: " + ", ".join(missing_deps))

    def read(self, src, side, src_dir=None):
        raise NotImplementedError()

    @staticmethod
    def _read_file(path):
        with codecs.open(path, "r", "utf-8") as f:
            for line in f:
                yield line
