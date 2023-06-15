# Code taken from bitsandbytes but modified with arg device to accept skipt_init
# from torch.nn.utils => makes model building way faster.
import os
import torch
import torch.nn as nn

try:
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    import bitsandbytes as bnb
except ImportError:
    raise ImportError("Install bitsandbytes to use 4/8bit compression")


class Linear4bit(nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype=None,
        compress_statistics=True,
        quant_type="fp4",
        device=torch.device("cpu"),
    ):
        super().__init__(input_features, output_features, bias)

        self.weight = bnb.nn.Params4bit(
            self.weight.data,
            requires_grad=False,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
        )
        self.compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor):
        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        if getattr(self.weight, "quant_state", None) is None:
            print(
                "FP4 quantization state not initialized. Please call .cuda() or"
                " .to(device) on the LinearFP4 layer first."
            )
        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        out = bnb.matmul_4bit(
            x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state
        )

        out = out.to(inp_dtype)

        return out


class Linear8bitLt(nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        has_fp16_weights=True,
        threshold=0.0,
        index=None,
        device=torch.device("cpu"),
    ):
        super().__init__(input_features, output_features, bias)
        self.state = bnb.MatmulLtState()
        self.index = index

        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = False
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = bnb.nn.Int8Params(
            self.weight.data,
            has_fp16_weights=has_fp16_weights,
            requires_grad=has_fp16_weights,
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if (
            not self.state.has_fp16_weights
            and self.state.CB is None
            and self.state.CxB is not None
        ):
            # reorder weight layout back from ampere/turing to row
            reorder_layout = True
            weight_clone = self.weight.data.clone()
        else:
            reorder_layout = False

        try:
            if reorder_layout:
                self.weight.data = bnb.autograd._functions.undo_layout(
                    self.state.CxB, self.state.tile_indices
                )

            super()._save_to_state_dict(destination, prefix, keep_vars)

            # we only need to save SCB as extra data, because CB for quantized weights
            # is already stored in weight.data
            weight_name = "SCB"

            # case 1: .cuda was called, SCB is in self.weight
            param_from_weight = getattr(self.weight, weight_name)
            # case 2: self.init_8bit_state was called, SCB is in self.state
            param_from_state = getattr(self.state, weight_name)

            key_name = prefix + f"{weight_name}"
            if param_from_weight is not None:
                destination[key_name] = (
                    param_from_weight if keep_vars else param_from_weight.detach()
                )
            elif not self.state.has_fp16_weights and param_from_state is not None:
                destination[key_name] = (
                    param_from_state if keep_vars else param_from_state.detach()
                )
        finally:
            if reorder_layout:
                self.weight.data = weight_clone

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for key in unexpected_keys:
            input_name = key[len(prefix) :]
            if input_name == "SCB":
                if self.weight.SCB is None:
                    # buffers not yet initialized, can't call them directly without
                    raise RuntimeError(
                        "Loading a quantized checkpoint into non-quantized Linear8bitLt is "
                        "not supported. Please call module.cuda() before module.load_state_dict()"
                    )

                input_param = state_dict[key]
                self.weight.SCB.copy_(input_param)
                unexpected_keys.remove(key)

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x: torch.Tensor):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        if not self.state.has_fp16_weights:
            if self.state.CB is not None and self.state.CxB is not None:
                # we converted 8-bit row major to turing/ampere format in the first inference pass
                # we no longer need the row-major weight
                del self.state.CB
                self.weight.data = self.state.CxB
        return out


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
                )
            elif q_type in ["bnb_FP4", "bnb_NF4"]:
                model._modules[name] = nn.utils.skip_init(
                    Linear4bit,
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    quant_type=q_type[-3:].lower(),  # 'fp4' or 'nf4'
                )
    return model
