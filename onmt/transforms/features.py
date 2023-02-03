from onmt.transforms import register_transform
from .transform import Transform
from onmt.utils.alignment import subword_map_by_joiner, subword_map_by_spacer
import re


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

    def _infer(self, example, side):
        if self.reversible_tokenization == "joiner":
            original_text = getattr(example, f"{side}_original") \
                if self.prior_tokenization else None
            word_to_subword_mapping = subword_map_by_joiner(
                getattr(example, side), original_subwords=original_text)
        else:  # Spacer
            word_to_subword_mapping = subword_map_by_spacer(
                getattr(example, side))

        new_feats = [[] for _ in range(len(getattr(example, f"{side}_feats")))]
        for subword, word_id in zip(
                getattr(example, side), word_to_subword_mapping):
            for i, feat_values in enumerate(getattr(example, f"{side}_feats")):
                # Punctuation only
                if not re.sub(r'(\W)+', '', subword).strip() \
                        and not self.prior_tokenization:
                    inferred_feat = "<null>"
                else:
                    inferred_feat = feat_values[word_id]
                new_feats[i].append(inferred_feat)
        setattr(example, f"{side}_feats", new_feats)

        # Security checks
        for feat in getattr(example, f"{side}_feats"):
            assert len(getattr(example, side)) == len(feat)

        return example

    def apply(self, example, is_train=False, stats=None, **kwargs):
        if example.src_feats is not None:
            example = self._infer(example, "src")

        if example.tgt_feats is not None:
            example = self._infer(example, "tgt")

        return example

    def _repr_args(self):
        return ''
