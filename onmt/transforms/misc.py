from onmt.utils.logging import logger
from onmt.transforms import register_transform
from .transform import Transform


@register_transform(name='filtertoolong')
class FilterTooLongTransform(Transform):
    """Filter out sentence that are too long."""

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to this Transform."""
        group = parser.add_argument_group("Transform/Filter")
        group.add("--src_seq_length", "-src_seq_length", type=int, default=200,
                  help="Maximum source sequence length.")
        group.add("--tgt_seq_length", "-tgt_seq_length", type=int, default=200,
                  help="Maximum target sequence length.")

    def _parse_opts(self):
        self.src_seq_length = self.opts.src_seq_length
        self.tgt_seq_length = self.opts.tgt_seq_length

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Return None if too long else return as is."""
        if (len(example['src']) > self.src_seq_length or
                len(example['tgt']) > self.tgt_seq_length):
            if stats is not None:
                stats.filter_too_long()
            return None
        else:
            return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}, {}={}'.format(
            'src_seq_length', self.src_seq_length,
            'tgt_seq_length', self.tgt_seq_length
        )


@register_transform(name='prefix')
class PrefixTransform(Transform):
    """Add Prefix to src (& tgt) sentence."""

    def __init__(self, opts):
        super().__init__(opts)

    @staticmethod
    def _get_prefix(corpus):
        """Get prefix string of a `corpus`."""
        if 'prefix' in corpus['transforms']:
            prefix = {
                'src': corpus['src_prefix'],
                'tgt': corpus['tgt_prefix']
            }
        else:
            prefix = None
        return prefix

    @classmethod
    def get_prefix_dict(cls, opts):
        """Get all needed prefix correspond to corpus in `opts`."""
        prefix_dict = {}
        for c_name, corpus in opts.data.items():
            prefix = cls._get_prefix(corpus)
            if prefix is not None:
                logger.info(f"Get prefix for {c_name}: {prefix}")
                prefix_dict[c_name] = prefix
        return prefix_dict

    @classmethod
    def get_specials(cls, opts):
        """Get special vocabs added by prefix transform."""
        prefix_dict = cls.get_prefix_dict(opts)
        src_specials, tgt_specials = set(), set()
        for _, prefix in prefix_dict.items():
            src_specials.update(prefix['src'].split())
            tgt_specials.update(prefix['tgt'].split())
        return (src_specials, tgt_specials)

    def warm_up(self, vocabs=None):
        """Warm up to get prefix dictionary."""
        super().warm_up(None)
        self.prefix_dict = self.get_prefix_dict(self.opts)

    def _prepend(self, example, prefix):
        """Prepend `prefix` to `tokens`."""
        for side, side_prefix in prefix.items():
            example[side] = side_prefix.split() + example[side]
        return example

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply prefix prepend to example.

        Should provide `corpus_name` to get correspond prefix.
        """
        corpus_name = kwargs.get('corpus_name', None)
        if corpus_name is None:
            raise ValueError('corpus_name is required.')
        corpus_prefix = self.prefix_dict.get(corpus_name, None)
        if corpus_prefix is None:
            raise ValueError(f'prefix for {corpus_name} does not exist.')
        return self._prepend(example, corpus_prefix)

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('prefix_dict', self.prefix_dict)
