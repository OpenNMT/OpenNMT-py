# coding: utf-8


class Datatype(object):
    """An abstraction for data types.

    Data types are meant to be singleton and instantiated
    on importing their module.

    Args:
        name (str)
        reader (class): A subclass of ``onmt.datatypes.DataReaderBase``.
        sort_key (Callable[[torchtext.data.Example], bool])
        fields (Callable[[str, ...],
            Tuple[List[Tuple[str, torchtext.data.Field]],
            List[Tuple[str, torchtext.data.Field]]]): A function that maps
            the base_name and any other kwargs to a list of tuples of
            fieldnames and fields to be at the top level and a list of
            tuples of fieldnames and fields to be at the "side" or "basename"
            level.

    """
    def __init__(self, name, reader, sort_key, fields):
        self.name = name
        self.reader = reader
        self.sort_key = sort_key
        self.fields = fields
