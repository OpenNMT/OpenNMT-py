# coding: utf-8


# several data readers need optional dependencies. There's no
# appropriate builtin exception
class MissingDependencyException(Exception):
    pass


class DataReaderBase(object):
    """Read data from file system and yield as dicts."""
    @classmethod
    def from_opt(cls, opt):
        return cls()

    @classmethod
    def _read_file(cls, path):
        with open(path, "rb") as f:
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
