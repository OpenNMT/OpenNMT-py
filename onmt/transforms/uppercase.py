from onmt.transforms import register_transform
from .transform import Transform, ObservableStats
import unicodedata
import random


@register_transform(name='uppercase')
class UpperCaseTransform(Transform):
    """Convert source and target examples to uppercase."""

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Avalailable options relate to this Transform."""

        group = parser.add_argument_group("Transform/Uppercase")
        group.add("--upper_corpus_ratio", "-upper_corpus_ratio", type=float,
                  default=0.01, help="Corpus ratio to apply uppercasing.")

    def _parse_opts(self):
        self.upper_corpus_ratio = self.opts.upper_corpus_ratio

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Convert source and target to uppercase."""

        if random.random() > self.upper_corpus_ratio:
            return example

        src_str = ' '.join(example['src'])
        tgt_str = ' '.join(example['tgt'])

        src_str = ''.join(c for c in unicodedata.normalize('NFD',
                          src_str.upper()) if unicodedata.category(c) != 'Mn')
        tgt_str = ''.join(c for c in unicodedata.normalize('NFD',
                          tgt_str.upper()) if unicodedata.category(c) != 'Mn')

        example['src'] = src_str.split()
        example['tgt'] = tgt_str.split()

        return example
