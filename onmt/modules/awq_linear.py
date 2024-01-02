import torch.nn as nn


def replace_awq_linear(
    model, module_to_convert=[], w_bit=4, group_size=128, q_type="llm_awq"
):
    if q_type in ["awq_gemm", "awq_gemv"]:
        try:
            from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
        except ImportError:
            raise ImportError("Install AutoAWQ to use awq")
        if q_type == "awq_gemm":
            AWQLin = WQLinear_GEMM
        else:
            AWQLin = WQLinear_GEMV
    else:
        raise ValueError("No Awq framework for this value")

    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_awq_linear(module, module_to_convert, w_bit, group_size, q_type)

        if isinstance(module, nn.Linear) and name in module_to_convert:
            model._modules[name] = AWQLin(
                w_bit=w_bit,
                group_size=group_size,
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                dev=module.weight.device,
            )
    return model
