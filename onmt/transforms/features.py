from onmt.utils.logging import logger
from onmt.transforms import register_transform
from .transform import Transform, ObservableStats
from onmt.constants import DefaultTokens, SubwordMarker
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
        super()._parse_opts()
        logger.info("Parsed pyonmttok kwargs for src: {}".format(
            self.opts.src_onmttok_kwargs))
        self.src_onmttok_kwargs = self.opts.src_onmttok_kwargs

    def apply(self, example, is_train=False, stats=None, **kwargs):

        if "src_feats" not in example:
            # Do nothing
            return example

        joiner = self.src_onmttok_kwargs["joiner"] if "joiner" in self.src_onmttok_kwargs else SubwordMarker.JOINER
        case_markup = SubwordMarker.CASE_MARKUP if "case_markup" in self.src_onmttok_kwargs else []
        # TODO: support joiner_new or spacer_new options. Consistency not ensured currently

        if "joiner_annotate" in self.src_onmttok_kwargs:
        	word_to_subword_mapping = subword_map_by_joiner(example["src"], marker=joiner, case_markup=case_markup)
        elif "spacer_annotate" in self.src_onmttok_kwargs:
        	# TODO: case markup
        	word_to_subword_mapping = subword_map_by_spacer(example["src"], marker=joiner)
       	else:
       		# TODO: support not reversible tokenization
       		raise Exception("InferFeats transform does not currently work without either joiner_annotate or spacer_annotate")

        inferred_feats = defaultdict(list)
        for subword, word_id in zip(example["src"], word_to_subword_mapping):
            for feat_name, feat_values in example["src_feats"].items():

                # If case markup placeholder
                if subword in case_markup:
                    inferred_feat = "<null>"

                # Punctuation only (assumes joiner is also some punctuation token)
                elif not re.sub(r'(\W)+', '', subword).strip():
                    inferred_feat = "<null>"

                else:
                    inferred_feat = feat_values[word_id]

                inferred_feats[feat_name].append(inferred_feat)

        for feat_name, feat_values in inferred_feats.items():
            example["src_feats"][feat_name] = inferred_feats[feat_name]

        return example

    def _repr_args(self):
        return ''