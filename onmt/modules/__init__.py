from onmt.modules.UtilClass import LayerNorm, Bottle, BottleLinear, \
    BottleLayerNorm, BottleSoftmax, Elementwise
from onmt.modules.Gate import ContextGateFactory
from onmt.modules.GlobalAttention import GlobalAttention
from onmt.modules.ConvMultiStepAttention import ConvMultiStepAttention
from onmt.modules.ImageEncoder import ImageEncoder
from onmt.modules.CopyGenerator import CopyGenerator, CopyCriterion
from onmt.modules.StructuredAttention import MatrixTree
from onmt.modules.Transformer import TransformerEncoder, TransformerDecoder
from onmt.modules.Conv2Conv import CNNEncoder, CNNDecoder
from onmt.modules.MultiHeadedAttn import MultiHeadedAttention
from onmt.modules.StackedRNN import StackedLSTM, StackedGRU
from onmt.modules.Embeddings import Embeddings
from onmt.modules.WeightNorm import WeightNormConv2d
from onmt.modules.Encoder import Encoder


# For flake8 compatibility.
__all__ = [GlobalAttention, ImageEncoder, CopyGenerator, MultiHeadedAttention,
           LayerNorm, Bottle, BottleLinear, BottleLayerNorm, BottleSoftmax,
           TransformerEncoder, TransformerDecoder, Embeddings, Elementwise,
           CopyCriterion, MatrixTree, WeightNormConv2d, ConvMultiStepAttention,
           CNNEncoder, CNNDecoder, StackedLSTM, StackedGRU, ContextGateFactory,
           Encoder]
