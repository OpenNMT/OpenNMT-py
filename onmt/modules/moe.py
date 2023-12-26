"""MoE mixture of experts"."""
import torch
import torch.nn as nn
from onmt.modules.position_ffn import PositionwiseFeedForward
from torch.distributed import all_reduce


class MoE(nn.Module):
    def __init__(
        self,
        num_experts,
        num_experts_per_tok,
        d_model,
        d_ff,
        dropout,
        pos_ffn_activation_fn,
        add_ffnbias,
        parallel_residual,
        layer_norm,
        norm_eps,
        use_ckpting=[],
        parallel_gpu=1,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                PositionwiseFeedForward(
                    d_model,
                    d_ff,
                    dropout,
                    pos_ffn_activation_fn,
                    add_ffnbias,
                    parallel_residual,
                    layer_norm,
                    norm_eps,
                    use_ckpting=use_ckpting,
                    parallel_gpu=parallel_gpu,
                )
                for i in range(num_experts)
            ]
        )
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.num_experts_per_tok = num_experts_per_tok
        self.parallel_gpu = parallel_gpu

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])

        scores = self.gate(x)
        if self.parallel_gpu > 1:
            all_reduce(scores)
        expert_weights, expert_indices = torch.topk(
            scores, self.num_experts_per_tok, dim=-1
        )
        expert_weights = expert_weights.softmax(dim=-1)
        flat_expert_indices = expert_indices.view(-1)

        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
        y = torch.empty_like(x)
        for i, expert in enumerate(self.experts):
            if torch.any(flat_expert_indices == i):
                y[flat_expert_indices == i] = expert(x[flat_expert_indices == i])
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(
            dim=1
        )
        return y.view(*orig_shape)
