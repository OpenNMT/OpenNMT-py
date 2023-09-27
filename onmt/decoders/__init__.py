"""Module defining decoders."""
import os
import importlib
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, StdRNNDecoder
from onmt.decoders.transformer import TransformerDecoder, TransformerLMDecoder
from onmt.decoders.cnn_decoder import CNNDecoder


str2dec = {
    "rnn": StdRNNDecoder,
    "ifrnn": InputFeedRNNDecoder,
    "cnn": CNNDecoder,
    "transformer": TransformerDecoder,
    "transformer_lm": TransformerLMDecoder,
}

__all__ = [
    "DecoderBase",
    "TransformerDecoder",
    "StdRNNDecoder",
    "CNNDecoder",
    "InputFeedRNNDecoder",
    "str2dec",
    "TransformerLMDecoder",
]


def get_decoders_cls(decoders_names):
    """Return valid encoder class indicated in `decoders_names`."""
    decoders_cls = {}
    for name in decoders_names:
        if name not in str2dec:
            raise ValueError("%s decoder not supported!" % name)
        decoders_cls[name] = str2dec[name]
    return decoders_cls


def register_decoder(name):
    """Encoder register that can be used to add new encoder class."""

    def register_decoder_cls(cls):
        if name in str2dec:
            raise ValueError("Cannot register duplicate decoder ({})".format(name))
        if not issubclass(cls, DecoderBase):
            raise ValueError(f"decoder ({name}: {cls.__name_}) must extend DecoderBase")
        str2dec[name] = cls
        __all__.append(cls.__name__)  # added to be complete
        return cls

    return register_decoder_cls


# Auto import python files in this directory
decoder_dir = os.path.dirname(__file__)
for file in os.listdir(decoder_dir):
    path = os.path.join(decoder_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        file_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("onmt.decoders." + file_name)
