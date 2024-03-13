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
            "--response_patterns",
            "-response_patterns",
            help="Response patten to locate the end of the prompt",
            default=["Response : ｟newline｠"],
            nargs="+",
        )

    def _parse_opts(self):
        self.response_patterns = self.opts.response_patterns

    def apply(self, example, is_train=False, stats=None, **kwargs):
        _src = " ".join(example["src"])
        response = None
        for _pattern in self.response_patterns:
            if len(_src.split(_pattern)) == 2:
                prompt, response = _src.split(_pattern)
                response = DefaultTokens.MASK_BEFORE.join([_pattern, response])
        if response is not None:
            _src = "".join([prompt, response])
            example["src"] = _src.split(" ")
            example["tgt"] = _src.split(" ")
        else:
            logger.info("The mask_before could not be inserted")
            return example
        return example
