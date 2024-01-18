from onmt.utils.logging import logger
from onmt.transforms import register_transform
from .transform import Transform, ObservableStats


class FilterTooLongStats(ObservableStats):
    """Runing statistics for FilterTooLongTransform."""

    __slots__ = ["filtered"]

    def __init__(self):
        self.filtered = 1

    def update(self, other: "FilterTooLongStats"):
        self.filtered += other.filtered


@register_transform(name="filtertoolong")
class FilterTooLongTransform(Transform):
    """Filter out sentence that are too long."""

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """
        Available options relate to this Transform.
        For performance it is better to use multiple of 8
        On target side, since we'll add BOS/EOS, we filter with minus 2
        """
        group = parser.add_argument_group("Transform/Filter")
        group.add(
            "--src_seq_length",
            "-src_seq_length",
            type=int,
            default=192,
            help="Maximum source sequence length.",
        )
        group.add(
            "--tgt_seq_length",
            "-tgt_seq_length",
            type=int,
            default=192,
            help="Maximum target sequence length.",
        )

    def _parse_opts(self):
        self.src_seq_length = self.opts.src_seq_length
        self.tgt_seq_length = self.opts.tgt_seq_length

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Return None if too long else return as is."""
        if (
            len(example["src"]) > self.src_seq_length
            or len(example["tgt"]) > self.tgt_seq_length - 2
        ):
            if stats is not None:
                stats.update(FilterTooLongStats())
            return None
        else:
            return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return "{}={}, {}={}".format(
            "src_seq_length", self.src_seq_length, "tgt_seq_length", self.tgt_seq_length
        )


@register_transform(name="prefix")
class PrefixTransform(Transform):
    """Add Prefix to src (& tgt) sentence."""

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Avalailable options relate to this Transform."""
        group = parser.add_argument_group("Transform/Prefix")
        group.add(
            "--src_prefix",
            "-src_prefix",
            type=str,
            default="",
            help="String to prepend to all source example.",
        )
        group.add(
            "--tgt_prefix",
            "-tgt_prefix",
            type=str,
            default="",
            help="String to prepend to all target example.",
        )

    @staticmethod
    def _get_prefix(corpus):
        """Get prefix string of a `corpus`."""
        if "prefix" in corpus["transforms"]:
            src_prefix = corpus.get("src_prefix", "")
            tgt_prefix = corpus.get("tgt_prefix", "")
            prefix = {"src": src_prefix, "tgt": tgt_prefix}
        else:
            prefix = None
        return prefix

    @classmethod
    def get_prefix_dict(cls, opts):
        """Get all needed prefix correspond to corpus in `opts`."""
        prefix_dict = {}
        # prefix src/tgt for each dataset
        if hasattr(opts, "data"):
            for c_name, corpus in opts.data.items():
                prefix = cls._get_prefix(corpus)
                if prefix is not None:
                    logger.debug(f"Get prefix for {c_name}: {prefix}")
                    prefix_dict[c_name] = prefix
        # prefix as general option for inference
        if hasattr(opts, "src_prefix"):
            if "infer" not in prefix_dict.keys():
                prefix_dict["infer"] = {}
            prefix_dict["infer"]["src"] = opts.src_prefix
            logger.debug(f"Get prefix for src infer: {opts.src_prefix}")
        if hasattr(opts, "tgt_prefix"):
            if "infer" not in prefix_dict.keys():
                prefix_dict["infer"] = {}
            prefix_dict["infer"]["tgt"] = opts.tgt_prefix
            logger.debug(f"Get prefix for tgt infer: {opts.tgt_prefix}")

        return prefix_dict

    @classmethod
    def get_specials(cls, opts):
        """Get special vocabs added by prefix transform."""
        prefix_dict = cls.get_prefix_dict(opts)
        src_specials, tgt_specials = set(), set()
        for _, prefix in prefix_dict.items():
            src_specials.update(prefix["src"].split(" "))
            tgt_specials.update(prefix["tgt"].split(" "))
        return (src_specials, tgt_specials)

    def warm_up(self, vocabs=None):
        """Warm up to get prefix dictionary."""
        super().warm_up(None)
        self.prefix_dict = self.get_prefix_dict(self.opts)

    def _prepend(self, example, prefix):
        """Prepend `prefix` to `tokens`."""
        for side, side_prefix in prefix.items():
            if example.get(side) is not None:
                example[side] = side_prefix.split(" ") + example[side]
            elif len(side_prefix) > 0:
                example[side] = side_prefix.split(" ")
        return example

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply prefix prepend to example.

        Should provide `corpus_name` to get correspond prefix.
        """
        corpus_name = kwargs.get("corpus_name", None)
        if corpus_name is None:
            raise ValueError("corpus_name is required.")
        corpus_prefix = self.prefix_dict.get(corpus_name, None)
        if corpus_prefix is None:
            raise ValueError(f"prefix for {corpus_name} does not exist.")
        return self._prepend(example, corpus_prefix)

    def apply_reverse(self, translated):
        def _removeprefix(s, prefix):
            if s.startswith(prefix) and len(prefix) > 0:
                return s[len(prefix) + 1 :]
            else:
                return s

        corpus_prefix = self.prefix_dict.get("infer", None)
        return _removeprefix(translated, corpus_prefix["tgt"])

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return "{}={}".format("prefix_dict", self.prefix_dict)


@register_transform(name="suffix")
class SuffixTransform(Transform):
    """Add Suffix to src (& tgt) sentence."""

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Avalailable options relate to this Transform."""
        group = parser.add_argument_group("Transform/Suffix")
        group.add(
            "--src_suffix",
            "-src_suffix",
            type=str,
            default="",
            help="String to append to all source example.",
        )
        group.add(
            "--tgt_suffix",
            "-tgt_suffix",
            type=str,
            default="",
            help="String to append to all target example.",
        )

    @staticmethod
    def _get_suffix(corpus):
        """Get suffix string of a `corpus`."""
        if "suffix" in corpus["transforms"]:
            src_suffix = corpus.get("src_suffix", "")
            tgt_suffix = corpus.get("tgt_suffix", "")
            suffix = {"src": src_suffix, "tgt": tgt_suffix}
        else:
            suffix = None
        return suffix

    @classmethod
    def get_suffix_dict(cls, opts):
        """Get all needed suffix correspond to corpus in `opts`."""
        suffix_dict = {}
        # suffix src/tgt for each dataset
        if hasattr(opts, "data"):
            for c_name, corpus in opts.data.items():
                suffix = cls._get_suffix(corpus)
                if suffix is not None:
                    logger.debug(f"Get suffix for {c_name}: {suffix}")
                    suffix_dict[c_name] = suffix
        # suffix as general option for inference
        if hasattr(opts, "src_suffix"):
            if "infer" not in suffix_dict.keys():
                suffix_dict["infer"] = {}
            suffix_dict["infer"]["src"] = opts.src_suffix
            logger.debug(f"Get suffix for src infer: {opts.src_suffix}")
        if hasattr(opts, "tgt_suffix"):
            if "infer" not in suffix_dict.keys():
                suffix_dict["infer"] = {}
            suffix_dict["infer"]["tgt"] = opts.tgt_suffix
            logger.debug(f"Get suffix for tgt infer: {opts.tgt_suffix}")

        return suffix_dict

    @classmethod
    def get_specials(cls, opts):
        """Get special vocabs added by suffix transform."""
        suffix_dict = cls.get_suffix_dict(opts)
        src_specials, tgt_specials = set(), set()
        for _, suffix in suffix_dict.items():
            src_specials.update(suffix["src"].split(" "))
            tgt_specials.update(suffix["tgt"].split(" "))
        return (src_specials, tgt_specials)

    def warm_up(self, vocabs=None):
        """Warm up to get suffix dictionary."""
        super().warm_up(None)
        self.suffix_dict = self.get_suffix_dict(self.opts)

    def _append(self, example, suffix):
        """Prepend `suffix` to `tokens`."""
        for side, side_suffix in suffix.items():
            if example.get(side) is not None:
                example[side] = example[side] + side_suffix.split(" ")
            elif len(side_suffix) > 0:
                example[side] = side_suffix.split(" ")
        return example

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply suffix append to example.

        Should provide `corpus_name` to get correspond suffix.
        """
        corpus_name = kwargs.get("corpus_name", None)
        if corpus_name is None:
            raise ValueError("corpus_name is required.")
        corpus_suffix = self.suffix_dict.get(corpus_name, None)
        if corpus_suffix is None:
            raise ValueError(f"suffix for {corpus_name} does not exist.")
        return self._append(example, corpus_suffix)

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return "{}={}".format("suffix_dict", self.suffix_dict)
