"""Module defining decoders."""
from onmt.decoders.cnn_decoder import CNNDecoder
from onmt.decoders.decoder import (
    DecoderBase,
    InputFeedRNNDecoder,
    StdRNNDecoder,
)
from onmt.decoders.transformer import (
    TransformerDecoder,
    TransformerLMDecoder,
    TransformerLMPseudoSelfAttentionDecoder,
)

str2dec = {
    "rnn": StdRNNDecoder,
    "ifrnn": InputFeedRNNDecoder,
    "cnn": CNNDecoder,
    "transformer": TransformerDecoder,
    "transformer_lm": TransformerLMDecoder,
    "transformer_lm_psa": TransformerLMPseudoSelfAttentionDecoder,
}

__all__ = [
    "DecoderBase",
    "TransformerDecoder",
    "StdRNNDecoder",
    "CNNDecoder",
    "InputFeedRNNDecoder",
    "str2dec",
    "TransformerLMDecoder",
    "TransformerLMPseudoSelfAttentionDecoder",
]
