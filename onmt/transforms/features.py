from onmt.transforms import register_transform
from .transform import Transform
from onmt.utils.alignment import subword_map_by_joiner, subword_map_by_spacer


@register_transform(name="inferfeats")
class InferFeatsTransform(Transform):
    """Infer features for subword tokenization."""

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Avalilable options related to this Transform."""
        group = parser.add_argument_group("Transform/InferFeats")
        group.add(
            "--reversible_tokenization",
            "-reversible_tokenization",
            default="joiner",
            choices=["joiner", "spacer"],
            help="Type of reversible tokenization " "applied on the tokenizer.",
        )

    def _parse_opts(self):
        super()._parse_opts()
        self.reversible_tokenization = self.opts.reversible_tokenization

    def apply(self, example, is_train=False, stats=None, **kwargs):
        if "src_feats" not in example:
            # Do nothing
            return example

        if self.reversible_tokenization == "joiner":
            original_src = example["src_original"]
            word_to_subword_mapping = subword_map_by_joiner(
                example["src"], original_subwords=original_src
            )
        else:  # Spacer
            word_to_subword_mapping = subword_map_by_spacer(example["src"])

        new_src_feats = [[] for _ in range(len(example["src_feats"]))]
        for subword, word_id in zip(example["src"], word_to_subword_mapping):
            for i, feat_values in enumerate(example["src_feats"]):
                inferred_feat = feat_values[word_id]
                new_src_feats[i].append(inferred_feat)
        example["src_feats"] = new_src_feats

        return example

    def _repr_args(self):
        return ""
