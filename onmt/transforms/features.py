from onmt.utils.logging import logger
from onmt.transforms import register_transform
from .transform import Transform, ObservableStats
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
                logger.warning(f"Skipping example due to mismatch between source and feature {feat_name}")
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
        pass

    def _parse_opts(self):
        pass

    def apply(self, example, is_train=False, stats=None, **kwargs):

        if "src_feats" not in example:
            # Do nothing
            return example

        feats_i = 0
        inferred_feats = defaultdict(list)   
        for subword in example["src"]:
            next_ = False
            for k, v in example["src_feats"].items():
                # TODO: what about custom placeholders??

                # Placeholders
                if re.match(r'｟\w+｠', subword):
                    inferred_feat = "N"

                # Punctuation only
                elif not re.sub(r'(\W)+', '', subword).strip():
                    inferred_feat = "N"

                # Joiner annotate
                elif re.search("￭", subword):
                    inferred_feat = v[feats_i]

                # Whole word
                else:
                    inferred_feat = v[feats_i]
                    next_ = True

                inferred_feats[k].append(inferred_feat)
            
            if next_:
                feats_i += 1

        # Check all features have been consumed
        for k, v in example["src_feats"].items(): 
        	assert feats_i == len(v), f'Not all features consumed for {k}'

        for k, v in inferred_feats.items():
            example["src_feats"][k] = inferred_feats[k]
        return example

    def _repr_args(self):
        return ''