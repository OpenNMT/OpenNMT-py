import torch.nn as nn
from awq.quantize.qmodule import WQLinear


def replace_awq_linear(
    model,
    module_to_convert=[],
    w_bit=4,
    group_size=128,
):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_awq_linear(module, module_to_convert, w_bit, group_size)

        if isinstance(module, nn.Linear) and name in module_to_convert:
            model._modules[name] = WQLinear(
                w_bit=w_bit,
                group_size=group_size,
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                dev=module.weight.device,
            )
    return model
