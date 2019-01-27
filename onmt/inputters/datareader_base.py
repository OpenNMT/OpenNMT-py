# coding: utf-8

import codecs


# several data readers need optional dependencies. There's no
# appropriate builtin exception
class MissingDependencyException(Exception):
    pass


class DataReaderBase(object):
    """Read data from file system and yield as dicts.

    Subclasses' ``__init__`` should take arbitrary kwargs. DataReaders
    are expected to take the union of all DataReader args and take
    only the ones they need. The rest should be ignored as
    ``**kwargs``.
    """
    def __init__(self, **kwargs):
        pass

    @classmethod
    def _read_file(cls, path):
        with codecs.open(path, "r", "utf-8") as f:
            for line in f:
                yield line

    @staticmethod
    def _raise_missing_dep(*missing_deps):
        """Raise missing dep exception with standard error message."""
        raise MissingDependencyException(
            "Could not create reader. Be sure to install "
            "the following dependencies: " + ", ".join(missing_deps))

    def read(self, data, side, src_dir):
        raise NotImplementedError()
