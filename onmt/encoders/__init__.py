"""Module defining encoders."""
import os
import importlib
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.ggnn_encoder import GGNNEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder


str2enc = {
    "ggnn": GGNNEncoder,
    "rnn": RNNEncoder,
    "brnn": RNNEncoder,
    "cnn": CNNEncoder,
    "transformer": TransformerEncoder,
    "mean": MeanEncoder,
}

__all__ = [
    "EncoderBase",
    "TransformerEncoder",
    "GGNNEncoder",
    "RNNEncoder",
    "CNNEncoder",
    "MeanEncoder",
    "str2enc",
]


def get_encoders_cls(encoder_names):
    """Return valid encoder class indicated in `encoder_names`."""
    encoders_cls = {}
    for name in encoder_names:
        if name not in str2enc:
            raise ValueError("%s encoder not supported!" % name)
        encoders_cls[name] = str2enc[name]
    return encoders_cls


def register_encoder(name):
    """Encoder register that can be used to add new encoder class."""

    def register_encoder_cls(cls):
        if name in str2enc:
            raise ValueError("Cannot register duplicate encoder ({})".format(name))
        if not issubclass(cls, EncoderBase):
            raise ValueError(f"encoder ({name}: {cls.__name_}) must extend EncoderBase")
        str2enc[name] = cls
        __all__.append(cls.__name__)  # added to be complete
        return cls

    return register_encoder_cls


# Auto import python files in this directory
encoder_dir = os.path.dirname(__file__)
for file in os.listdir(encoder_dir):
    path = os.path.join(encoder_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        file_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("onmt.encoders." + file_name)
