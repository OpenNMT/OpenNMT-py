from onmt.utils.logging import logger
from onmt.transforms import register_transform
from .transform import Transform
from onmt.utils.alignment import subword_map_by_joiner, subword_map_by_spacer
import re
from collections import defaultdict


@register_transform(name='filterfeats')
class FilterFeatsTransform(Transform):
    """Filter out examples with a mismatch between source and features."""

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        pass

    def _parse_opts(self):
        pass

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Return None if mismatch"""

        if 'src_feats' not in example:
            # Do nothing
            return example

        for feat_name, feat_values in example['src_feats'].items():
            if len(example['src']) != len(feat_values):
                logger.warning(
                    f"Skipping example due to mismatch "
                    f"between source and feature {feat_name}")
                return None
        return example

    def _repr_args(self):
        return ''


@register_transform(name='inferfeats')
class InferFeatsTransform(Transform):
    """Infer features for subword tokenization."""

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Avalilable options related to this Transform."""
        group = parser.add_argument_group("Transform/InferFeats")
        group.add("--reversible_tokenization", "-reversible_tokenization",
                  default="joiner", choices=["joiner", "spacer"],
                  help="Type of reversible tokenization "
                       "applied on the tokenizer.")
        group.add("--prior_tokenization", "-prior_tokenization",
                  default=False, action="store_true",
                  help="Whether the input has already been tokenized.")

    def _parse_opts(self):
        super()._parse_opts()
        self.reversible_tokenization = self.opts.reversible_tokenization
        self.prior_tokenization = self.opts.prior_tokenization

    def apply(self, example, is_train=False, stats=None, **kwargs):

        if "src_feats" not in example:
            # Do nothing
            return example

        if self.reversible_tokenization == "joiner":
            original_src = example["src_original"] \
                if self.prior_tokenization else None
            word_to_subword_mapping = subword_map_by_joiner(
                example["src"], original_subwords=original_src)
        else:  # Spacer
            word_to_subword_mapping = subword_map_by_spacer(example["src"])

        inferred_feats = defaultdict(list)
        for subword, word_id in zip(example["src"], word_to_subword_mapping):
            for feat_name, feat_values in example["src_feats"].items():
                # Punctuation only
                if not re.sub(r'(\W)+', '', subword).strip() \
                        and not self.prior_tokenization:
                    inferred_feat = "<null>"
                else:
                    inferred_feat = feat_values[word_id]

                inferred_feats[feat_name].append(inferred_feat)

        for feat_name, feat_values in inferred_feats.items():
            example["src_feats"][feat_name] = inferred_feats[feat_name]

        return example

    def _repr_args(self):
        return ''
