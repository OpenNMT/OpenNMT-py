"""
Implementation of "Attention is All You Need" and of
subsequent transformer based architectures
"""

import torch
import torch.nn as nn

from onmt.decoders.decoder import DecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction
from onmt.utils.misc import sequence_mask


class TransformerDecoderLayerBase(nn.Module):
    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        self_attn_type="scaled-dot",
        max_relative_positions=0,
        aan_useffn=False,
        full_context_alignment=False,
        alignment_heads=0,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False
    ):
        """
        Args:
            d_model (int): the dimension of keys/values/queries in
                :class:`MultiHeadedAttention`, also the input size of
                the first-layer of the :class:`PositionwiseFeedForward`.
            heads (int): the number of heads for MultiHeadedAttention.
            d_ff (int): the second-layer of the
                :class:`PositionwiseFeedForward`.
            dropout (float): dropout in residual, self-attn(dot) and
                feed-forward
            attention_dropout (float): dropout in context_attn  (and
                self-attn(avg))
            self_attn_type (string): type of self-attention scaled-dot,
                average
            max_relative_positions (int):
                Max distance between inputs in relative positions
                representations
            aan_useffn (bool): Turn on the FFN layer in the AAN decoder
            full_context_alignment (bool):
                whether enable an extra full context decoder forward for
                alignment
            alignment_heads (int):
                N. of cross attention heads to use for alignment guiding
            pos_ffn_activation_fn (ActivationFunction):
                activation function choice for PositionwiseFeedForward layer
            add_qkvbias (bool): whether to add bias to the Key/Value nn.Linear

        """
        super(TransformerDecoderLayerBase, self).__init__()

        self.self_attn_type = self_attn_type
        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads,
                d_model,
                dropout=attention_dropout,
                max_relative_positions=max_relative_positions,
                attn_type="self",
                add_qkvbias=add_qkvbias
            )
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(
                d_model, dropout=attention_dropout, aan_useffn=aan_useffn
            )

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout,
                                                    pos_ffn_activation_fn
                                                    )
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.full_context_alignment = full_context_alignment
        self.alignment_heads = alignment_heads

    def forward(self, *args, **kwargs):
        """Extend `_forward` for (possibly) multiple decoder pass:
        Always a default (future masked) decoder forward pass,
        Possibly a second future aware decoder pass for joint learn
        full context alignement, :cite:`garg2019jointly`.

        Args:
            * All arguments of _forward.
            with_align (bool): whether return alignment attention.

        Returns:
            (FloatTensor, FloatTensor, FloatTensor or None):

            * layer_out ``(batch_size, T, model_dim)``
            * top_attn ``(batch_size, T, src_len)``
            * attn_align ``(batch_size, T, src_len)`` or None
        """
        with_align = kwargs.pop("with_align", False)
        layer_out, attns = self._forward(*args, **kwargs)
        top_attn = attns[:, 0, :, :].contiguous()
        attn_align = None
        if with_align:
            if self.full_context_alignment:
                # return _, (B, Q_len, K_len)
                _, attns = self._forward(*args, **kwargs, future=True)

            if self.alignment_heads > 0:
                attns = attns[:, : self.alignment_heads, :, :].contiguous()
            # layer average attention across heads, get ``(B, Q, K)``
            # Case 1: no full_context, no align heads -> layer avg baseline
            # Case 2: no full_context, 1 align heads -> guided align
            # Case 3: full_context, 1 align heads -> full cte guided align
            attn_align = attns.mean(dim=1)
        return layer_out, top_attn, attn_align

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_dec_mask(self, tgt_pad_mask, future):
        tgt_len = tgt_pad_mask.size(-1)
        if not future:  # apply future_mask, result mask in (B, T, T)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8,
            )
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            # BoolTensor was introduced in pytorch 1.2
            try:
                future_mask = future_mask.bool()
            except AttributeError:
                pass
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
        else:  # only mask padding, result mask in (B, 1, T)
            dec_mask = tgt_pad_mask
        return dec_mask

    def _forward_self_attn(self, layer_in_norm, dec_mask, step):
        if self.self_attn_type == "scaled-dot":
            return self.self_attn(
                layer_in_norm,
                layer_in_norm,
                layer_in_norm,
                mask=dec_mask
            )
        elif self.self_attn_type == "average":
            return self.self_attn(
                layer_in_norm, mask=dec_mask, step=step
            )
        else:
            raise ValueError(
                f"self attention {type(self.self_attn)} not supported"
            )


class TransformerDecoderLayer(TransformerDecoderLayerBase):
    """Transformer Decoder layer block in Pre-Norm style.
    Pre-Norm style is an improvement w.r.t. Original paper's Post-Norm style,
    providing better converge speed and performance. This is also the actual
    implementation in tensor2tensor and also avalable in fairseq.
    See https://tunz.kr/post/4 and :cite:`DeeperTransformer`.

    """

    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        self_attn_type="scaled-dot",
        max_relative_positions=0,
        aan_useffn=False,
        full_context_alignment=False,
        alignment_heads=0,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False
    ):
        """
        Args:
            See TransformerDecoderLayerBase
        """
        super(TransformerDecoderLayer, self).__init__(
            d_model,
            heads,
            d_ff,
            dropout,
            attention_dropout,
            self_attn_type,
            max_relative_positions,
            aan_useffn,
            full_context_alignment,
            alignment_heads,
            pos_ffn_activation_fn=pos_ffn_activation_fn,
            add_qkvbias=add_qkvbias
        )
        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            attn_type="context",
            add_qkvbias=add_qkvbias
        )
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

    def update_dropout(self, dropout, attention_dropout):
        super(TransformerDecoderLayer, self).update_dropout(
            dropout, attention_dropout
        )
        self.context_attn.update_dropout(attention_dropout)

    def _forward(
        self,
        layer_in,
        enc_out,
        src_pad_mask,
        tgt_pad_mask,
        step=None,
        future=False,
    ):
        """A naive forward pass for transformer decoder.

        # T: could be 1 in the case of stepwise decoding or tgt_len

        Args:
            layer_in (FloatTensor): ``(batch_size, T, model_dim)``
            enc_out (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (bool): ``(batch_size, 1, src_len)``
            tgt_pad_mask (bool): ``(batch_size, 1, T)``
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.

        Returns:
            (FloatTensor, FloatTensor):

            * layer_out ``(batch_size, T, model_dim)``
            * attns ``(batch_size, head, T, src_len)``

        """
        dec_mask = None
        src_pad_mask = src_pad_mask.unsqueeze(1)  # [B,1,1,slen]

        if layer_in.size(1) > 1:
            # masking is necessary when sequence length is greater than one
            dec_mask = self._compute_dec_mask(tgt_pad_mask, future)
            dec_mask = dec_mask.unsqueeze(1)
            dec_mask = dec_mask.expand(-1, -1, dec_mask.size(3), -1)
            src_pad_mask = src_pad_mask.expand(-1, -1, dec_mask.size(3), -1)
            # mask now are (batch x 1 x tlen x s or t len)
            # 1 = heads to be expanded in MHA

        layer_in_norm = self.layer_norm_1(layer_in)

        query, _ = self._forward_self_attn(
            layer_in_norm, dec_mask, step
        )

        query = self.drop(query) + layer_in

        query_norm = self.layer_norm_2(query)

        mid, attns = self.context_attn(
            enc_out,
            enc_out,
            query_norm,
            mask=src_pad_mask
        )
        layer_out = self.feed_forward(self.drop(mid) + query)

        return layer_out, attns


class TransformerDecoderBase(DecoderBase):
    def __init__(self, d_model, copy_attn, embeddings, alignment_layer):
        super(TransformerDecoderBase, self).__init__()

        self.embeddings = embeddings

        # Decoder State
        self.state = {}

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.alignment_layer = alignment_layer

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_hid_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0]
            if type(opt.attention_dropout) is list
            else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            opt.aan_useffn,
            opt.full_context_alignment,
            opt.alignment_layer,
            alignment_heads=opt.alignment_heads,
            pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
            add_qkvbias=opt.add_qkvbias
        )

    def init_state(self, src, enc_out, enc_final_hs):
        """Initialize decoder state."""
        self.state["src"] = src

    def map_state(self, fn):

        if self.state["src"] is not None:
            self.state["src"] = fn(self.state["src"], 0)
        for layer in self.transformer_layers:
            if hasattr(layer, 'context_attn'):
                if layer.context_attn.layer_cache[1]['keys'].numel() != 0:
                    x = fn(layer.context_attn.layer_cache[1]['keys'], 0)
                    y = fn(layer.context_attn.layer_cache[1]['values'], 0)
                    layer.context_attn.layer_cache = True, {'keys': x,
                                                            'values': y}
            if isinstance(layer.self_attn, AverageAttention):
                if layer.self_attn.layer_cache[1]['prev_g'].numel() != 0:
                    x = fn(layer.self_attn.layer_cache[1]['prev_g'], 0)
                    layer.self_attn.layer_cache = True, {'prev_g': x}
            else:
                if layer.self_attn.layer_cache[1]['keys'].numel() != 0:
                    x = fn(layer.self_attn.layer_cache[1]['keys'], 0)
                    y = fn(layer.self_attn.layer_cache[1]['values'], 0)
                    layer.self_attn.layer_cache = True, {'keys': x,
                                                         'values': y}

    def detach_state(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)


class TransformerDecoder(TransformerDecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
        num_layers (int): number of decoder layers.
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        copy_attn (bool): if using a separate copy attention
        self_attn_type (str): type of self-attention scaled-dot, average
        dropout (float): dropout in residual, self-attn(dot) and feed-forward
        attention_dropout (float): dropout in context_attn (and self-attn(avg))
        embeddings (onmt.modules.Embeddings):
            embeddings to use, should have positional encodings
        max_relative_positions (int):
            Max distance between inputs in relative positions representations
        aan_useffn (bool): Turn on the FFN layer in the AAN decoder
        full_context_alignment (bool):
            whether enable an extra full context decoder forward for alignment
        alignment_layer (int): N° Layer to supervise with for alignment guiding
        alignment_heads (int):
            N. of cross attention heads to use for alignment guiding
        add_qkvbias (bool): whether to add bias to the Key/Value nn.Linear
    """

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        copy_attn,
        self_attn_type,
        dropout,
        attention_dropout,
        embeddings,
        max_relative_positions,
        aan_useffn,
        full_context_alignment,
        alignment_layer,
        alignment_heads,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False
    ):
        super(TransformerDecoder, self).__init__(
            d_model, copy_attn, embeddings, alignment_layer
        )

        self.transformer_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    self_attn_type=self_attn_type,
                    max_relative_positions=max_relative_positions,
                    aan_useffn=aan_useffn,
                    full_context_alignment=full_context_alignment,
                    alignment_heads=alignment_heads,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                    add_qkvbias=add_qkvbias
                )
                for i in range(num_layers)
            ]
        )

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, enc_out=None, step=None, **kwargs):
        """
        Decode, possibly stepwise.
        when training step is always None, when decoding, step increases
        tgt (Tensor): batch x tlen x feats
        enc_out (Tensor): encoder output (batch x slen x model_dim)
        """
        if enc_out is None:
            enc_out = self.embeddings(tgt)
        if step == 0:
            self._init_cache(enc_out)

        tgt_words = tgt[:, :, 0]

        emb = self.embeddings(tgt, step=step)
        dec_out = emb
        assert emb.dim() == 3  # len x batch x embedding_dim

        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["src_len"]
        src_max_len = self.state["src"].shape[1]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len)  # [B x slen]
        src_pad_mask = src_pad_mask.unsqueeze(1)  # [B x 1 x slen]
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        with_align = kwargs.pop("with_align", False)
        attn_aligns = []

        for layer in self.transformer_layers:
            dec_out, attn, attn_align = layer(
                dec_out,
                enc_out,
                src_pad_mask,
                tgt_pad_mask,
                step=step,
                with_align=with_align,
            )
            if attn_align is not None:
                attn_aligns.append(attn_align)

        dec_out = self.layer_norm(dec_out)

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
            # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_out, attns

    def _init_cache(self, enc_out):

        batch_size = enc_out.size(0)
        depth = enc_out.size(-1)

        for layer in self.transformer_layers:
            # first value set to True triggered by the beginning of decoding
            # layer_cache becomes active in the MultiHeadedAttention fwd
            layer.context_attn.layer_cache = (
                True,
                {'keys': torch.tensor([], device=enc_out.device),
                 'values': torch.tensor([], device=enc_out.device)}
                )
            if isinstance(layer.self_attn, AverageAttention):
                layer.self_attn.layer_cache = True, {'prev_g': torch.zeros(
                     (batch_size, 1, depth), device=enc_out.device
                ).to(enc_out.dtype)}
            else:
                layer.self_attn.layer_cache = (
                    True,
                    {'keys': torch.tensor([], device=enc_out.device),
                     'values': torch.tensor([], device=enc_out.device)}
                    )


class TransformerLMDecoderLayer(TransformerDecoderLayerBase):
    """Transformer Decoder only layer block in GPT style.
   Args:
        See TransformerDecoderLayerBase
    """

    def _forward(
        self, layer_in, tgt_pad_mask, step=None, future=False
    ):
        """A naive forward pass for transformer decoder.

        # T: could be 1 in the case of stepwise decoding or tgt_len

        Args:
            layer_in (FloatTensor): ``(batch_size, T, model_dim)``
            tgt_pad_mask (bool): ``(batch_size, 1, T)``
            layer_cache (dict or None): cached layer info when stepwise decode
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.

        Returns:
            (FloatTensor, FloatTensor):

            * layer_out ``(batch_size, T, model_dim)``
            * attns ``(batch_size, head, T, T)``

        """
        dec_mask = None

        if layer_in.size(1) > 1:
            # masking is necessary when sequence length is greater than one
            dec_mask = self._compute_dec_mask(tgt_pad_mask, future)
            dec_mask = dec_mask.unsqueeze(1)
            dec_mask = dec_mask.expand(-1, -1, dec_mask.size(3), -1)
            # mask now are (batch x 1 x tlen x tlen)
            # 1 = heads to be expanded in MHA

        layer_in_norm = self.layer_norm_1(layer_in)

        query, attns = self._forward_self_attn(
            layer_in_norm, dec_mask, step
        )

        layer_out = self.drop(query) + layer_in

        layer_out = self.feed_forward(layer_out)

        return layer_out, attns


class TransformerLMDecoder(TransformerDecoderBase):
    """The Transformer decoder from GPT-2
   Args:
        num_layers (int): number of decoder layers.
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        copy_attn (bool): if using a separate copy attention
        self_attn_type (str): type of self-attention scaled-dot, average
        dropout (float): dropout in residual, self-attn(dot) and feed-forward
        attention_dropout (float): dropout in context_attn (and self-attn(avg))
        embeddings (onmt.modules.Embeddings):
            embeddings to use, should have positional encodings
        max_relative_positions (int):
            Max distance between inputs in relative positions representations
        aan_useffn (bool): Turn on the FFN layer in the AAN decoder
        add_qkvbias (bool): whether to add bias to the Key/Value nn.Linear
    """

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        copy_attn,
        self_attn_type,
        dropout,
        attention_dropout,
        embeddings,
        max_relative_positions,
        aan_useffn,
        full_context_alignment=None,
        alignment_layer=None,
        alignment_heads=None,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False
    ):
        super(TransformerLMDecoder, self).__init__(
            d_model, copy_attn, embeddings, None
        )
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLMDecoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    self_attn_type=self_attn_type,
                    max_relative_positions=max_relative_positions,
                    aan_useffn=aan_useffn,
                    full_context_alignment=None,
                    alignment_heads=None,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                    add_qkvbias=add_qkvbias
                )
                for i in range(num_layers)
            ]
        )

    def init_state(self, src=None, enc_out=None, enc_final_hs=None):
        super(TransformerLMDecoder, self).init_state(None, None, None)

    def detach_state(self):
        pass

    def forward(self, tgt, enc_out=None, step=None, **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(tgt)

        tgt_words = tgt[:, :, 0]

        dec_out = self.embeddings(tgt, step=step)
        assert dec_out.dim() == 3  # batch x len x embedding_dim

        pad_idx = self.embeddings.word_padding_idx
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        with_align = kwargs.pop("with_align", False)
        assert not with_align, "TransformerLMDecoder does not support align"

        for layer in self.transformer_layers:
            dec_out, attn, _ = layer(
                dec_out,
                tgt_pad_mask,
                step=step,
                with_align=with_align,
            )

        dec_out = self.layer_norm(dec_out)

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_out, attns

    def _init_cache(self, tgt=None):

        for layer in self.transformer_layers:
            if isinstance(layer.self_attn, AverageAttention):
                raise NotImplementedError
            else:
                layer.self_attn.layer_cache = (
                    True,
                    {'keys': torch.tensor([], device=tgt.device),
                     'values': torch.tensor([], device=tgt.device)}
                    )
