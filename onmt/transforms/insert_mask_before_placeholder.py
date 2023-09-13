from onmt.constants import DefaultTokens
from onmt.transforms import register_transform
from onmt.utils.logging import logger
from .transform import Transform


@register_transform(name="insert_mask_before_placeholder")
class InsertMaskBeforePlaceholdersTransform(Transform):
    """Add the `DefaultTokens.MASK_BEFORE` placeholder between
    the prompt and the response in an LM finetuning exemple.
    This is necessary to enable the 'zero-out prompt loss' mechanism.
    """

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Options for mask_before placeholders insertion"""

        group = parser.add_argument_group(
            "Transform/InsertMaskBeforePlaceholdersTransform"
        )
        group.add(
            "--response_pattern",
            "-response_pattern",
            type=str,
            help="Response patten to locate the end of the prompt",
            default="Response : ｟newline｠",
        )

    def _parse_opts(self):
        self.response_pattern = self.opts.response_pattern

    def apply(self, example, is_train=False, stats=None, **kwargs):
        _src = " ".join(example["src"])
        if len(_src.split(self.response_pattern)) != 2:
            logger.info("The mask_before could not be inserted")
            return example
        prompt, response = _src.split(self.response_pattern)
        response = DefaultTokens.MASK_BEFORE.join([self.response_pattern, response])
        _src = "".join([prompt, response])
        example["src"] = _src.split(" ")
        example["tgt"] = _src.split(" ")
        return example
