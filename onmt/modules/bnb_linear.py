# Code taken from bitsandbytes but modified with arg device to accept skipt_init
# from torch.nn.utils => makes model building way faster.
import os
import torch
import torch.nn as nn

try:
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    from bitsandbytes import MatmulLtState
    from bitsandbytes.nn import (
        Linear4bit,
        Linear8bitLt,
        Params4bit,
        Int8Params,
    )
except ImportError:
    raise ImportError("Install bitsandbytes to use 4/8bit compression")


def replace_bnb_linear(
    model,
    module_to_convert=[],
    q_type="bnb_8bit",
    threshold=6.0,
    compute_dtype=torch.float16,  # we could also use bfloat16 when available
):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_bnb_linear(
                module, module_to_convert, q_type, threshold, compute_dtype
            )

        if isinstance(module, nn.Linear) and name in module_to_convert:
            if q_type == "bnb_8bit":
                model._modules[name] = nn.utils.skip_init(
                    Linear8bitLt,
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    has_fp16_weights=False,
                    threshold=threshold,
                    device=torch.device("cpu"),
                )
                model._modules[name].state = MatmulLtState()
                model._modules[name].index = None
                model._modules[name].state.threshold = threshold
                model._modules[name].state.has_fp16_weights = False
                model._modules[name].state.memory_efficient_backward = False
                if threshold > 0.0:
                    model._modules[name].state.use_pool = True
                model._modules[name].weight = Int8Params(
                    model._modules[name].weight.data,
                    has_fp16_weights=False,
                    requires_grad=False,
                )
            elif q_type in ["bnb_FP4", "bnb_NF4"]:
                model._modules[name] = nn.utils.skip_init(
                    Linear4bit,
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    quant_type=q_type[-3:].lower(),  # 'fp4' or 'nf4'
                    device=torch.device("cpu"),
                )
                model._modules[name].weight = Params4bit(
                    model._modules[name].weight.data,
                    requires_grad=False,
                    quant_type=q_type[-3:].lower(),
                )
                model._modules[name].compute_dtype = compute_dtype
    return model
