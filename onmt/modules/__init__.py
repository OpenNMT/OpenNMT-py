def aeq(*args):
    base = args[0]
    for a in args[1:]:
        assert a == base, str(args)

from onmt.modules.GlobalAttention import GlobalAttention
from onmt.modules.ImageEncoder import ImageEncoder
from onmt.modules.CopyGenerator import CopyGenerator, CopyCriterion
from onmt.modules.StructuredAttention import MatrixTree
from onmt.modules.Transformer import TransformerEncoder, TransformerDecoder
from onmt.modules.MultiHeadedAttn import MultiHeadedAttention
from onmt.modules.StackedRNN import StackedLSTM, StackedGRU
from onmt.modules.Util import LayerNorm, Bottle, BottleLinear, \
    BottleLayerNorm, BottleSoftmax


# For flake8 compatibility.
__all__ = [GlobalAttention, ImageEncoder, CopyGenerator, MultiHeadedAttention,
           LayerNorm, Bottle, BottleLinear, BottleLayerNorm, BottleSoftmax,
           TransformerEncoder, TransformerDecoder,
           CopyCriterion, MatrixTree, StackedLSTM, StackedGRU, aeq]
