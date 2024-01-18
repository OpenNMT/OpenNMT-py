from onmt.transforms import register_transform
from .transform import Transform
import unicodedata
import random


@register_transform(name="uppercase")
class UpperCaseTransform(Transform):
    """
    Convert source and target examples to uppercase.

    This transform uses `unicodedata` to normalize the converted
    uppercase strings as this is needed for some languages (e.g. Greek).
    One issue is that the normalization removes all diacritics and
    accents from the uppercased strings, even though in few occasions some
    diacritics should be kept even in the uppercased form.
    """

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Add an option for the corpus ratio to apply this transform."""

        group = parser.add_argument_group("Transform/Uppercase")
        group.add(
            "--upper_corpus_ratio",
            "-upper_corpus_ratio",
            type=float,
            default=0.01,
            help="Corpus ratio to apply uppercasing.",
        )

    def _parse_opts(self):
        self.upper_corpus_ratio = self.opts.upper_corpus_ratio

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Convert source and target examples to uppercase."""

        if random.random() > self.upper_corpus_ratio:
            return example

        src_str = " ".join(example["src"])
        src_str = "".join(
            c
            for c in unicodedata.normalize("NFD", src_str.upper())
            if unicodedata.category(c) != "Mn"
        )
        example["src"] = src_str.split(" ")

        if example["tgt"] is not None:
            tgt_str = " ".join(example["tgt"])
            tgt_str = "".join(
                c
                for c in unicodedata.normalize("NFD", tgt_str.upper())
                if unicodedata.category(c) != "Mn"
            )
            example["tgt"] = tgt_str.split(" ")

        return example
