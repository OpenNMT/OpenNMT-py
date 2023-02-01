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

    def apply(self, example, is_train=False, stats=None, **kwargs):

        if "src_feats" not in example:
            # Do nothing
            assert False, "TODO"
            return example

        if self.reversible_tokenization == "joiner":
            original_src = example["src_original"] \
                if self.prior_tokenization else None
            word_to_subword_mapping = subword_map_by_joiner(
                example["src"], original_subwords=original_src)
        else:  # Spacer
            word_to_subword_mapping = subword_map_by_spacer(example["src"])

        new_src_feats = [[] for _ in range(len(example["src_feats"]))]
        for subword, word_id in zip(example["src"], word_to_subword_mapping):
            for i, feat_values in enumerate(example["src_feats"]):
                # Punctuation only
                if not re.sub(r'(\W)+', '', subword).strip() \
                        and not self.prior_tokenization:
                    inferred_feat = "<null>"
                else:
                    inferred_feat = feat_values[word_id]
                new_src_feats[i].append(inferred_feat)
        example["src_feats"] = new_src_feats

        # Security checks
        for feat in example["src_feats"]:
            assert len(example["src"]) == len(feat)

        if self.reversible_tokenization == "joiner":
            original_tgt = example["tgt_original"] \
                if self.prior_tokenization else None
            word_to_subword_mapping = subword_map_by_joiner(
                example["tgt"], original_subwords=original_tgt)
        else:  # Spacer
            word_to_subword_mapping = subword_map_by_spacer(example["tgt"])

        new_tgt_feats = [[] for _ in range(len(example["tgt_feats"]))]
        for subword, word_id in zip(example["tgt"], word_to_subword_mapping):
            for i, feat_values in enumerate(example["tgt_feats"]):
                # Punctuation only
                if not re.sub(r'(\W)+', '', subword).strip() \
                        and not self.prior_tokenization:
                    inferred_feat = "<null>"
                else:
                    inferred_feat = feat_values[word_id]
                new_tgt_feats[i].append(inferred_feat)
        example["tgt_feats"] = new_tgt_feats

        # Security checks
        for feat in example["tgt_feats"]:
            assert len(example["tgt"]) == len(feat)

        return example

    def _repr_args(self):
        return ''
