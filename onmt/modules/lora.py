#  --------------------------------------------------------------------------
#  copied and adapted https://github.com/microsoft/LoRA/
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT).
# Support bnb quantization of nderlying layers
#  --------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from typing import Dict
import os


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            merge_weights=merge_weights,
        )
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default
            # for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(
                        0, 1
                    ) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(
                        0, 1
                    ) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x,
                    self.lora_A.transpose(0, 1),
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class QLinear(type):
    def __call__(cls, *args, **kwargs):
        quant_type = kwargs.get("quant_type", None)
        use_ckpting = kwargs.get("use_ckpting", [])
        r = kwargs.get("r", 0)
        lora_alpha = kwargs.get("lora_alpha", 1)
        lora_dropout = kwargs.get("lora_dropout", 0.0)
        bias = kwargs.get("bias", False)
        merge_weights = kwargs.get("merge_weights", True)
        threshold = kwargs.get("threshold", 6.0)
        compute_dtype = kwargs.get("compute_dtype", torch.float16)

        if quant_type in ["bnb_8bit", "bnb_FP4", "bnb_NF4"]:
            try:
                os.environ["BITSANDBYTES_NOWELCOME"] = "1"
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("Install bitsandbytes to use 4/8bit compression")

        if quant_type == "bnb_8bit":
            layer_class = bnb.nn.Linear8bitLt
        elif quant_type in ["bnb_FP4", "bnb_NF4"]:
            layer_class = bnb.nn.Linear4bit
        else:
            layer_class = nn.Linear

        class QLoraLinear_cls(layer_class, LoRALayer):
            def __init__(self, *args, **kwargs):
                if quant_type == "bnb_8bit":
                    super(QLoraLinear_cls, self).__init__(
                        *args, bias=bias, has_fp16_weights=False, threshold=threshold
                    )
                elif quant_type in ["bnb_FP4", "bnb_NF4"]:
                    super(QLoraLinear_cls, self).__init__(
                        *args,
                        bias=bias,
                        compute_dtype=compute_dtype,
                        quant_type=quant_type[-3:].lower(),
                    )
                else:
                    super(QLoraLinear_cls, self).__init__(*args, bias=bias)
                self.quant_type = quant_type
                LoRALayer.__init__(self, r, lora_alpha, lora_dropout, merge_weights)
                # Actual trainable parameters
                if r > 0:
                    self.lora_A = nn.Parameter(self.weight.new_zeros((r, args[0])))
                    self.lora_B = nn.Parameter(self.weight.new_zeros((args[1], r)))
                    self.scaling = self.lora_alpha / self.r
                    # Freezing the pre-trained weight matrix
                    self.weight.requires_grad = False
                self.reset_parameters()
                self.maybe_ckpt = (
                    checkpoint if "lora" in use_ckpting else lambda f, x: f(x)
                )

            def reset_parameters(self):
                # we do not super().reset_parameters() save lot of time and useless when no grad.
                if hasattr(self, "lora_A"):
                    # initialize A the same way as the default
                    # for nn.Linear and B to zero
                    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                    nn.init.zeros_(self.lora_B)

            def train(self, mode: bool = True):
                if self.quant_type is None:
                    super().train(mode)
                    if mode:
                        if self.merge_weights and self.merged:
                            # Make sure that the weights are not merged
                            if self.r > 0:
                                self.weight.data -= (
                                    self.lora_B @ self.lora_A
                                ) * self.scaling
                            self.merged = False
                    else:
                        if self.merge_weights and not self.merged:
                            # Merge the weights and mark it
                            if self.r > 0:
                                self.weight.data += (
                                    self.lora_B @ self.lora_A
                                ) * self.scaling
                            self.merged = True
                else:
                    # cannot merge/unmerge quantized weigts with unquantized lora_X
                    pass

            def forward(self, x: torch.Tensor):
                if self.r > 0 and not self.merged:
                    result = (
                        self.maybe_ckpt(super().forward, x)
                        + (
                            self.lora_dropout(x)
                            @ self.lora_A.transpose(0, 1)
                            @ self.lora_B.transpose(0, 1)
                        )
                        * self.scaling
                    )
                else:
                    result = self.maybe_ckpt(super().forward, x)
                return result

        instance = QLoraLinear_cls.__new__(
            QLoraLinear_cls
        )  # Create a new instance of QLoraLinear_cls
        instance.__init__(
            *args, **kwargs
        )  # Invoke the __init__ method of QLoraLinear_cls

        """
        # Check if QLoraLinear has a custom __init__ method
        if hasattr(cls, "__init__"):
            kwargs.pop("r", None)
            kwargs.pop("quant_type", None)
            # Invoke the __init__ method of QLoraLinear
            cls.__init__(instance, *args, **kwargs)
        """
        return instance


class QLoraLinear(metaclass=QLinear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        **kwargs,
    ):
        pass


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = "none") -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {
            k: my_state_dict[k] for k in my_state_dict if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def replace_lora_linear(
    model,
    r=2,
    lora_alpha=1,
    lora_dropout=0,
    layer="",
    quant_type=None,
    use_ckpting=[],
    threshold=6.0,
    compute_dtype=torch.float16,
):
    """
    Function replacing layers with LoRa layers recursively.
    Args:
        model:
        r: rank of matrix of the Low Rank layer
        lora_alpha: cf paper
        lora_dropout: cf paper
        layer: layer name of the model to be replaced
        quant_type: use bnb to quantize nn.Linear sub-layer
    """
    for name, module in model.named_children():
        if hasattr(module, "children") and len(list(module.children())) > 0:
            replace_lora_linear(
                module,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                layer=layer,
                quant_type=quant_type,
                use_ckpting=use_ckpting,
                threshold=threshold,
                compute_dtype=compute_dtype,
            )

        if isinstance(module, nn.Linear) and name == layer:
            model._modules[name] = QLoraLinear(
                module.in_features,
                module.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=module.bias is not None,
                quant_type=quant_type,
                use_ckpting=use_ckpting,
                threshold=threshold,
                compute_dtype=compute_dtype,
            )
    return model


def replace_lora_embedding(model, r=2, lora_alpha=1):
    """
    Function replacing Embeddings with LoRa ones recursively.
    Args:
        model:
        r: rank of matrix of the Low Rank layer
        lora_alpha: cf paper
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_lora_embedding(module, r, lora_alpha)
        if isinstance(module, nn.Embedding):
            model._modules[name] = Embedding(
                module.num_embeddings,
                module.embedding_dim,
                r=r,
                lora_alpha=lora_alpha,
                padding_idx=module.padding_idx,
                sparse=module.sparse,
            )
    return model
