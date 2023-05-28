"""  Attention and normalization modules  """
import importlib
import os
from onmt.modules.util_class import Elementwise
from onmt.modules.gate import context_gate_factory, ContextGate
from onmt.modules.global_attention import GlobalAttention
from onmt.modules.conv_multi_step_attention import ConvMultiStepAttention
from onmt.modules.copy_generator import CopyGenerator, CopyGeneratorLoss
from onmt.modules.multi_headed_attn import MultiHeadedAttention
from onmt.modules.embeddings import Embeddings, PositionalEncoding
from onmt.modules.weight_norm import WeightNormConv2d
from onmt.modules.average_attn import AverageAttention
from onmt.modules.alibi_position_bias import AlibiPositionalBias
from onmt.modules.lora import LoRALayer, Embedding, LoraLinear
from onmt.modules.lora import mark_only_lora_as_trainable, lora_state_dict
from onmt.modules.rmsnorm import RMSNorm

if importlib.util.find_spec("bitsandbytes") is not None:
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    import bitsandbytes as bnb
    from onmt.modules.lora import LoraLinear8bit, LoraLinear4bit

__all__ = [
    "Elementwise",
    "context_gate_factory",
    "ContextGate",
    "GlobalAttention",
    "ConvMultiStepAttention",
    "CopyGenerator",
    "CopyGeneratorLoss",
    "CopyGeneratorLMLossCompute",
    "MultiHeadedAttention",
    "Embeddings",
    "PositionalEncoding",
    "AlibiPositionalBias",
    "WeightNormConv2d",
    "AverageAttention",
    "RMSNorm",
    "LoRALayer",
    "Embedding",
    "LoraLinear",
    "LoraLinear8bit",
    "LoraLinear4bit",
    "mark_only_lora_as_trainable",
    "lora_state_dict",
]
