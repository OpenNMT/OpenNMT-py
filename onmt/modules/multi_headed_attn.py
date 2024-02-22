""" Multi-Head Attention module """
import torch
import torch.nn as nn
from math import log, sqrt
from torch import Tensor
from typing import Optional, Tuple
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.checkpoint import checkpoint
from torch.nn.utils import skip_init
from .alibi_position_bias import AlibiPositionalBias
from torch.distributed import all_reduce
from importlib import import_module

# Help functions for Rotary Embeddings
# https://arxiv.org/pdf/2104.09864.pdf
# too convoluted to make maxseqlen a parameter.
# we suppose src_seq_len at training and max_length at inference
# are both < 2048 tokens.


def rotaryembeddings(dim: int, maxseqlen=2048, base=10000, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    tmax = torch.arange(maxseqlen, device=inv_freq.device)
    rope = torch.outer(tmax, inv_freq).float()
    # rope is now matrix [maxseqlen, dim/2]
    rope = torch.polar(torch.ones_like(rope), rope)
    rope = torch.cat((rope, rope), dim=1)
    if device is not None:
        rope = rope.to(device)
    cos = rope[:, : rope.size(1) // 2].real.contiguous().half()
    sin = rope[:, : rope.size(1) // 2].imag.contiguous().half()
    return rope, cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(query, key, rope, interleave):
    if interleave:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        query_ = query.float().reshape(*query.shape[:-1], -1, 2)
        query_ = torch.view_as_complex(query_)
        key_ = key.float().reshape(*key.shape[:-1], -1, 2)
        key_ = torch.view_as_complex(key_)
        rope = rope[:, : rope.size(1) // 2].view(1, query_.size(1), 1, query_.size(3))
        query_out = torch.view_as_real(query_ * rope).flatten(3)
        key_out = torch.view_as_real(key_ * rope).flatten(3)
        return query_out.transpose(1, 2).type_as(query), key_out.transpose(
            1, 2
        ).type_as(key)
    else:
        cos, sin = rope.real, rope.imag
        rotary_dim = cos.size(1)
        head_dim = query.size(3)
        if rotary_dim < head_dim:
            q_embed = (query[:, :, :, :rotary_dim] * cos) + (
                rotate_half(query[:, :, :, :rotary_dim]) * sin
            )
            k_embed = (key[:, :, :, :rotary_dim] * cos) + (
                rotate_half(key[:, :, :, :rotary_dim]) * sin
            )
            q_embed = torch.cat([q_embed, query[:, :, :, rotary_dim:]], dim=-1)
            k_embed = torch.cat([k_embed, key[:, :, :, rotary_dim:]], dim=-1)
        else:
            q_embed = (query * cos) + (rotate_half(query) * sin)
            k_embed = (key * cos) + (rotate_half(key) * sin)
        return q_embed.type_as(query), k_embed.type_as(key)


# Help functions for max_relative positions
# https://arxiv.org/abs/1803.02155


def relative_matmul(x: Tensor, z: Tensor, transpose: bool) -> Tensor:
    """
    Helper function for relative positions attention.
    https://arxiv.org/pdf/1803.02155.pdf
    x shape [batch_size x heads x q_len x k_len]
    """
    batch_size, heads, length, _ = x.size()
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.contiguous().view(length, heads * batch_size, -1)
    if transpose:
        z = z.transpose(1, 2)
    x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.view(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t


def gen_relative_positions(
    length: int,
    max_relative_positions: int,
    cache: bool = False,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Generate the clipped relative positions matrix
    for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length + 1, 1, 1, device=device).unsqueeze(0)
    else:
        range_vec = torch.arange(length, device=device)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(
        distance_mat, min=-max_relative_positions, max=max_relative_positions
    )
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat


def _relative_position_bucket(
    relative_position, bidirectional=True, num_buckets=32, max_distance=128
):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/
    mesh_tensorflow/transformer/transformer_layers.py#L593
    Translate relative position to a bucket number for relative attention.
    The relative position is defined as memory_position - query_position,
    i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid.
    We use smaller buckets for small absolute relative_position and larger buckets for
    larger absolute relative_positions. All relative positions >=max_distance map to the
    same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the
    model has been trained on

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values
        in the range [0, num_buckets)
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(
            relative_position, torch.zeros_like(relative_position)
        )
    # now relative_position is in the range [0, inf)
    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions
    # up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(
        is_small, relative_position, relative_position_if_large
    )
    return relative_buckets


def compute_bias(
    query_length,
    key_length,
    is_decoder,
    max_relative_positions,
    relative_positions_buckets,
    device=None,
):
    """Compute binned relative position bias"""
    context_position = torch.arange(query_length, dtype=torch.long, device=device)[
        :, None
    ]
    memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    relative_position = (
        memory_position - context_position
    )  # shape (query_length, key_length)
    relative_position_bucket = _relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=(not is_decoder),
        num_buckets=relative_positions_buckets,
        max_distance=max_relative_positions,
    )
    return relative_position_bucket


# Help functions to split model dim per head


def shape(x: Tensor, dim_per_head: int) -> Tensor:
    """
    Projection.
    [batchsize x length x modeldim]
    -> [batchsize x heads x length x dimperhead]
    """
    x_0, x_1, _ = x.size()
    return x.view(x_0, x_1, -1, dim_per_head).transpose(1, 2)


def unshape(x: Tensor) -> Tensor:
    """
    Compute context.
    [batchsize x heads x length x dimperhead]
    -> [batchsize x length x modeldim]
    """
    x_0, x_1, _, x_3 = x.size()
    return x.transpose(1, 2).contiguous().view(x_0, -1, x_1 * x_3)


class MultiHeadedAttention(torch.nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
       max_relative_positions (int): max relative positions
       attn_type: "self" or "context"
    """

    def __init__(
        self,
        head_count: int,
        model_dim: int,
        dropout: float = 0.1,
        is_decoder: bool = True,
        max_relative_positions: int = 0,
        relative_positions_buckets: int = 0,
        rotary_interleave: bool = True,
        rotary_theta: int = 1e4,
        rotary_dim: int = 0,
        attn_type: str = None,
        self_attn_type: str = None,
        add_qkvbias=False,
        num_kv=0,
        use_ckpting=[],
        parallel_gpu=1,
    ) -> None:
        assert (
            model_dim % head_count == 0
        ), "Model dimension must be divisible by the number of heads"
        self.dim_per_head = model_dim // head_count
        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count
        self.num_kv = num_kv
        self.parallel_gpu = parallel_gpu
        if num_kv == 0:
            assert (
                model_dim % parallel_gpu == 0
            ), "Model dimension must be divisible by the number of partitions"
            self.linear_keys = skip_init(
                nn.Linear,
                in_features=model_dim,
                out_features=model_dim // parallel_gpu,
                bias=add_qkvbias,
            )
            self.linear_values = skip_init(
                nn.Linear,
                in_features=model_dim,
                out_features=model_dim // parallel_gpu,
                bias=add_qkvbias,
            )
        else:
            assert (
                self.dim_per_head * self.num_kv
            ) % parallel_gpu == 0, (
                "Model dimension must be divisible by the number of partitions"
            )
            self.linear_keys = skip_init(
                nn.Linear,
                in_features=model_dim,
                out_features=self.dim_per_head * self.num_kv // parallel_gpu,
                bias=add_qkvbias,
            )
            self.linear_values = skip_init(
                nn.Linear,
                in_features=model_dim,
                out_features=self.dim_per_head * self.num_kv // parallel_gpu,
                bias=add_qkvbias,
            )
        self.linear_query = skip_init(
            nn.Linear,
            in_features=model_dim,
            out_features=model_dim // parallel_gpu,
            bias=add_qkvbias,
        )
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout
        self.final_linear = skip_init(
            nn.Linear,
            in_features=model_dim // parallel_gpu,
            out_features=model_dim,
            bias=add_qkvbias,
        )
        self.is_decoder = is_decoder
        self.max_relative_positions = max_relative_positions
        self.relative_positions_buckets = relative_positions_buckets
        self.attn_type = attn_type
        self.self_attn_type = self_attn_type
        self.layer_cache = (
            False,
            {"keys": torch.tensor([]), "values": torch.tensor([])},
        )
        if relative_positions_buckets > 0:
            self.relative_attention_bias = nn.Embedding(
                relative_positions_buckets, head_count
            )
            self.relative_positions_embeddings = None
        elif max_relative_positions > 0:
            # https://arxiv.org/pdf/1803.02155.pdf
            # in the paper they suggest either two embeds
            # relative_key / relative_value or only
            # relative_key. We implemented the same embed
            # for both.
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head
            )
            self.relative_attention_bias = None
        else:
            self.relative_positions_embeddings = None
            self.relative_attention_bias = None

            if max_relative_positions == -1:  # rotary embeddings
                if rotary_dim == 0:
                    self.rotary_dim = self.dim_per_head
                else:
                    self.rotary_dim = rotary_dim
                self.rope, self.cos, self.sin = rotaryembeddings(
                    self.rotary_dim, base=rotary_theta
                )
                self.rotary_interleave = rotary_interleave
                self.rotary_theta = rotary_theta
            else:
                self.cos = None
                self.sin = None
                self.rotary_interleave = None
            if max_relative_positions == -2:  # alibi positional bias
                self.alibi = AlibiPositionalBias(head_count)

        self.maybe_ckpt = checkpoint if "mha" in use_ckpting else lambda f, x: f(x)

        try:
            flash_pack = import_module("flash_attn")
            if (
                hasattr(flash_pack, "flash_attn_func")
                and torch.cuda.get_device_capability()[0] >= 8
            ):
                self.flash_attn_func = getattr(flash_pack, "flash_attn_func")
                self.flash_attn_with_kvcache = getattr(
                    flash_pack, "flash_attn_with_kvcache"
                )
                self.flash2 = True
            else:
                self.flash2 = False
        except ImportError:
            self.flash2 = False

    def update_dropout(self, dropout: float) -> None:
        self.dropout.p = dropout
        self.dropout_p = dropout

    def forward(
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        mask: Optional[Tensor] = None,
        sliding_window: Optional[int] = 0,
        step: Optional[int] = 0,
        return_attn: Optional[bool] = False,
        self_attn_type: str = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the context vector and the attention vectors.

        Args:
           key (Tensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (Tensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (Tensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
           step (int): decoding step (used for Rotary embedding)
        Returns:
           (Tensor, Tensor):

           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """
        # 1) Project key, value, and query.
        # as a reminder at training layer_cache[0] remains False
        key_pad_mask = self.layer_cache[1].get("key_pad_mask", None)
        if self.layer_cache[0]:
            # Retrieve keys and values from the KV cache (decoding mode only).
            if self.attn_type == "self":
                query, key, value = (
                    self.linear_query(query),
                    self.linear_keys(query),
                    self.linear_values(query),
                )

                query = shape(query, self.dim_per_head)
                key = shape(key, self.dim_per_head)
                value = shape(value, self.dim_per_head)
                start_pos = step
                seqlen = query.size(2)

                if (
                    step == 0
                    or not self.flash2
                    or self.self_attn_type != "scaled-dot-flash"
                    or self.max_relative_positions not in [0, -1]
                    or query.size(0) > 128
                    or query.dtype != torch.float16
                ):
                    if self.max_relative_positions == -1:  # Rotary Embeddings
                        if seqlen + start_pos > self.rope.size(0):
                            # Resize rotary embeddings.
                            self.rope, _, _ = rotaryembeddings(
                                self.rotary_dim,
                                maxseqlen=(seqlen + start_pos + 2048),
                                base=self.rotary_theta,
                                device=self.rope.device,
                            )
                        rope = self.rope[start_pos : start_pos + seqlen]
                        query, key = apply_rotary_emb(
                            query, key, rope, interleave=self.rotary_interleave
                        )

                    if self.layer_cache[1]["keys"].numel() != 0:
                        key = torch.cat((self.layer_cache[1]["keys"], key), dim=2)
                        value = torch.cat((self.layer_cache[1]["values"], value), dim=2)
                        if sliding_window > 0 and key.size(2) > sliding_window:
                            key = key[:, :, 1:, :]
                            value = value[:, :, 1:, :]

                    self.layer_cache[1]["keys"] = key
                    self.layer_cache[1]["values"] = value

                else:
                    if start_pos >= self.layer_cache[1]["keys"].size(2):
                        self.layer_cache[1]["keys"] = torch.cat(
                            [
                                self.layer_cache[1]["keys"],
                                torch.zeros(
                                    self.layer_cache[1]["keys"].shape[:-2]
                                    + (32,)
                                    + self.layer_cache[1]["keys"].shape[-1:],
                                    device=query.device,
                                ).half(),
                            ],
                            dim=-2,
                        )
                        self.layer_cache[1]["values"] = torch.cat(
                            [
                                self.layer_cache[1]["values"],
                                torch.zeros(
                                    self.layer_cache[1]["values"].shape[:-2]
                                    + (32,)
                                    + self.layer_cache[1]["values"].shape[-1:],
                                    device=query.device,
                                ).half(),
                            ],
                            dim=-2,
                        )
                        if (
                            self.max_relative_positions == -1
                            and start_pos + 32 >= self.rope.size(0)
                        ):
                            # Resize rotary embeddings.
                            # We take a margin of 32 tokens as the kv_cache
                            #   is incremented by 32 tokens every 32 tokens.
                            self.rope, self.cos, self.sin = rotaryembeddings(
                                self.rotary_dim,
                                maxseqlen=(start_pos + 2048),
                                base=self.rotary_theta,
                                device=self.rope.device,
                            )

                    if sliding_window > 0 and key.size(2) > sliding_window:
                        self.layer_cache[1]["keys"] = self.layer_cache[1]["keys"][
                            :, :, 1:, :
                        ]
                        self.layer_cache[1]["values"] = self.layer_cache[1]["values"][
                            :, :, 1:, :
                        ]
                    context = self.flash_attn_with_kvcache(
                        query.transpose(1, 2),
                        self.layer_cache[1]["keys"].transpose(1, 2),
                        self.layer_cache[1]["values"].transpose(1, 2),
                        key.transpose(1, 2),
                        value.transpose(1, 2),
                        rotary_cos=self.cos,
                        rotary_sin=self.sin,
                        cache_seqlens=step,
                        rotary_interleaved=self.rotary_interleave,
                    ).transpose(1, 2)
                    attn_output = self.final_linear(unshape(context))
                    if self.parallel_gpu > 1:
                        all_reduce(attn_output)
                    return attn_output, None

            elif self.attn_type == "context":
                query = self.linear_query(query)
                query = shape(query, self.dim_per_head)
                if self.layer_cache[1]["keys"].numel() == 0:
                    key, value = self.linear_keys(key), self.linear_values(value)
                    key = shape(key, self.dim_per_head)
                    value = shape(value, self.dim_per_head)
                else:
                    key, value = (
                        self.layer_cache[1]["keys"],
                        self.layer_cache[1]["values"],
                    )
                self.layer_cache[1]["keys"] = key
                self.layer_cache[1]["values"] = value

            if key_pad_mask is not None:
                # Increase the cached key pad mask by concatenation.
                # For decoding only.
                if step > 0:
                    y = torch.zeros(
                        (key_pad_mask.size(0), key_pad_mask.size(1), 1),
                        dtype=torch.bool,
                        device=key_pad_mask.device,
                    )
                    self.layer_cache[1]["key_pad_mask"] = torch.cat(
                        (key_pad_mask, y), 2
                    )
                    key_pad_mask = self.layer_cache[1]["key_pad_mask"]
        else:
            # Retrieve keys and values from linear layers (training mode).
            key = self.maybe_ckpt(self.linear_keys, key)
            value = self.maybe_ckpt(self.linear_values, value)
            query = self.maybe_ckpt(self.linear_query, query)

            key = shape(key, self.dim_per_head)
            value = shape(value, self.dim_per_head)
            query = shape(query, self.dim_per_head)

            if self.max_relative_positions == -1:  # Rotary Embeddings
                start_pos = 0
                seqlen = query.size(2)
                if seqlen > self.rope.size(0):
                    # Resize rotary embeddings.
                    self.rope, self.cos, self.sin = rotaryembeddings(
                        self.rotary_dim,
                        maxseqlen=(seqlen + 2048),
                        base=self.rotary_theta,
                        device=query.device,
                    )
                rope = self.rope[start_pos : start_pos + seqlen]
                query, key = apply_rotary_emb(
                    query, key, rope, interleave=self.rotary_interleave
                )

        b, h, l, d = key.size()
        if self.num_kv > 0:
            qh = query.size(1)
            # expand key on heads dimension when it's less than query heads (multi-query variant)
            key = key.view(b, -1, 1, l, d).repeat(1, 1, qh // h, 1, 1)
            key = key.view(b, qh, l, d)

            # expand value on heads dimension when it's less than query heads (multi-query variant)
            value = value.view(b, -1, 1, l, d).repeat(1, 1, qh // h, 1, 1)
            value = value.view(b, qh, l, d)

        # 2) When standard pos. enc. or rotary, use flash attention

        # Ultimately flashv2 will be part of pytorch https://github.com/pytorch/pytorch/pull/105602
        # In the meantime: if vanilla tranformer or Rotary embeddings (not rel_pos, not alibi)
        # then use flash2 if seq len > 256 otherwise use xtransformer from pt2 uptream
        flash2 = (
            self.flash2
            and l > 256  # https://github.com/Dao-AILab/flash-attention/issues/591
        )

        if (
            self.max_relative_positions in [-1, 0]
            and not return_attn
            and query.device != torch.device("cpu")
            and self.self_attn_type == "scaled-dot-flash"
        ):
            # Apply flash2 attention.
            causal = self.is_decoder and self.attn_type == "self" and mask is not None
            if self.is_decoder and self.attn_type == "self" and flash2:
                if causal:
                    window_size = (
                        (-1, -1) if sliding_window == 0 else (sliding_window, 0)
                    )
                else:
                    window_size = (-1, -1)
                attn_output = self.flash_attn_func(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    dropout_p=self.dropout_p,
                    causal=causal,
                    window_size=window_size,
                ).transpose(1, 2)
            else:
                # Apply scaled dot product attention.
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=True
                ):
                    attn_output = scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        ~mask if mask is not None else None,
                        self.dropout_p,
                        is_causal=causal,
                    )
            attn = None

        else:
            query /= sqrt(self.dim_per_head)
            # batch x num_heads x query_len x key_len
            scores = torch.matmul(query, key.transpose(2, 3))

            if self.relative_attention_bias is not None:
                q_len = key.size(2) if self.layer_cache[0] else query.size(2)
                relative_position_bucket = compute_bias(
                    q_len,
                    key.size(2),
                    self.is_decoder,
                    self.max_relative_positions,
                    self.relative_positions_buckets,
                    device=key.device,
                )
                values = self.relative_attention_bias(
                    relative_position_bucket
                )  # shape (query_length, key_length, num_heads)
                position_bias = values.permute([2, 0, 1]).unsqueeze(
                    0
                )  # shape (1, num_heads, query_length, key_length)
                if self.layer_cache[0]:
                    position_bias = position_bias[:, :, -query.size(2) :, :]
                scores.add_(position_bias)

            elif self.relative_positions_embeddings is not None:
                key_len = key.size(2)
                # 1 or key_len x key_len
                relative_positions_matrix = gen_relative_positions(
                    key_len,
                    self.max_relative_positions,
                    cache=self.layer_cache[0],
                    device=key.device,
                )
                #  1 or key_len x key_len x dim_per_head
                relations_keys = self.relative_positions_embeddings(
                    relative_positions_matrix
                )
                scores.add_(relative_matmul(query, relations_keys, True))
            elif self.max_relative_positions == -2:  # Alibi
                scores = self.alibi(scores)

            scores = scores.float()
            if key_pad_mask is not None and mask is None:
                mask = key_pad_mask.unsqueeze(1)

            if mask is not None:
                # not 100% necessary but expand to nb of heads
                mask = mask.expand(-1, self.head_count // self.parallel_gpu, -1, -1)
                # now mask and scores have the same shape
                scores = scores.masked_fill(mask, -1e18)

            # 3) Apply attention dropout and compute context vectors.
            attn = self.softmax(scores).to(query.dtype)
            drop_attn = self.dropout(attn) if self.dropout_p > 0 else attn

            attn_output = torch.matmul(drop_attn, value)

            if self.relative_positions_embeddings is not None:
                # We use the same embeddings for key and value
                relations_values = relations_keys
                attn_output.add_(relative_matmul(drop_attn, relations_values, False))

        context = unshape(attn_output)
        if key_pad_mask is not None:
            if key_pad_mask.size(0) > 1 and context.size(1) > 1:
                x = key_pad_mask.squeeze(1).unsqueeze(2).expand(-1, -1, context.size(2))
                context = context.masked_fill(x, 0)

        if self.layer_cache[0]:
            attn_output = self.final_linear(context)
        else:
            attn_output = self.maybe_ckpt(self.final_linear, context)

        if self.parallel_gpu > 1:
            all_reduce(attn_output)

        return attn_output, attn
