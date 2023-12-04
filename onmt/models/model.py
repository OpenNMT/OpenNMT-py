""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn
from glob import glob


class BaseModel(nn.Module):
    """Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder / decoder or decoder only model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object"""

    def __init__(self, encoder, decoder):
        super(BaseModel, self).__init__()

    def forward(self, src, tgt, src_len, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.

        Args:
            src (Tensor): A source sequence passed to encoder.
                Typically for input this will be a padded `LongTensor`
                of size ``(batch, len, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(batch, tgt_len, features)``.
            src_len(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If bptt is false then init decoder state.
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(batch, tgt_len, hidden)``
            * dictionary of attention weights ``(batch, tgt_len, src_len)``"""

        raise NotImplementedError

    def update_dropout(self, dropout, attention_dropout):
        raise NotImplementedError

    def count_parameters(self, log=print):
        raise NotImplementedError

    def _load_param(self, name, module, param_name, param, buf_list, ckpt_t, offset):
        if module.__class__.__name__ == "WQLinear_GEMM":
            # ugly patch because in_feat and out_feat are reversed in WQLinear_GEMM
            param.data = param.data.transpose(0, 1)
            ckpt_t = ckpt_t.transpose(0, 1)
        if name.split(".")[-1] in [
            "linear_keys",
            "linear_values",
            "linear_query",
            "w_1",
            "w_3",
        ]:
            col_slice_start = param.data.size(0) * offset
            col_slice_end = param.data.size(0) * (offset + 1)
        else:
            col_slice_start = 0
            col_slice_end = param.data.size(0)
        if param.data.dim() == 2:
            if name.split(".")[-1] in ["final_linear", "w_2"]:
                row_slice_start = param.data.size(1) * offset
                row_slice_end = param.data.size(1) * (offset + 1)
            else:
                row_slice_start = 0
                row_slice_end = param.data.size(1)
            assert (
                param.data.size()
                == ckpt_t[
                    col_slice_start:col_slice_end,
                    row_slice_start:row_slice_end,
                ].size()
            ), "An error in model's partition and checkpoint's slice was detected"
            if name + "." + param_name in buf_list:
                if module.__class__.__name__ == "WQLinear_GEMM":
                    module.register_buffer(
                        param_name,
                        ckpt_t[
                            col_slice_start:col_slice_end,
                            row_slice_start:row_slice_end,
                        ].transpose(0, 1),
                    )
                else:
                    module.register_buffer(
                        param_name,
                        ckpt_t[
                            col_slice_start:col_slice_end,
                            row_slice_start:row_slice_end,
                        ],
                    )
            else:
                param.data = ckpt_t[
                    col_slice_start:col_slice_end,
                    row_slice_start:row_slice_end,
                ]
        else:
            assert (
                param.data.size() == ckpt_t[col_slice_start:col_slice_end].size()
            ), "An error in model's partition and checkpoint's slice was detected"
            if name + "." + param_name in buf_list:
                module.register_buffer(
                    param_name, ckpt_t[col_slice_start:col_slice_end]
                )
            else:
                param.data = ckpt_t[col_slice_start:col_slice_end]

    def load_state_dict(
        self,
        checkpoint,
        precision=torch.float32,
        device=torch.device("cpu"),
        strict=True,
        offset=0,
    ):
        """Custom state_dict loading to enable moving module on device as they are loaded

        Args:
            checkpoint: Pytorch serialized checkpoint
            precision: precision to move each module to
            device: device to move each module to
            strict: if True checks model keys wrt state_dict (both ways)
        """

        # bitsandbytes quantize weights when .cuda() is called
        # for huge models we need to save Ram
        # so we load the weights  module by module and transfer them to GPU for quantization
        if device == torch.device("cpu"):
            offset = 0
        buf_list = []
        for buf_name, buf in self.named_buffers():
            buf_list.append(buf_name)
        for name, module in self.named_modules():
            named_buf_and_param = list(module.named_buffers()) + list(
                module.named_parameters()
            )
            for param_name, param in named_buf_and_param:
                if len(param_name.split(".")) == 1:  # only last key
                    if name + "." + param_name in checkpoint["model"].keys():
                        ckpt_t = checkpoint["model"][name + "." + param_name]
                        self._load_param(
                            name, module, param_name, param, buf_list, ckpt_t, offset
                        )
                        del checkpoint["model"][name + "." + param_name]
                    elif (
                        "generator" in checkpoint.keys()
                        and "generator" in name
                        and checkpoint["generator"] is not None
                        and param_name in checkpoint["generator"].keys()
                    ):
                        keyname = (
                            name + "." + param_name if "linear" in name else param_name
                        )
                        param.data = checkpoint["generator"][keyname]
                        del checkpoint["generator"][keyname]
                    elif strict and "lora" not in param_name:
                        raise ValueError(
                            "Missing key in checkpoint: %s" % name + "." + param_name
                        )
                    if precision != torch.int8:
                        module.to(precision)
                    module.to(device)

        for key in checkpoint[
            "model"
        ].keys():  # if some keys are left in checkpoint after deletion
            if key not in buf_list:
                raise ValueError(
                    "Extra keys in model state_dict do not match the model config %s"
                    % checkpoint["model"].keys()
                )
        if checkpoint["generator"]:
            for key in checkpoint["generator"].keys():
                if key not in buf_list:
                    raise ValueError(
                        "Extra keys in generator state_dict do not match the model config %s"
                        % checkpoint["generator"].keys()
                    )

    def load_safe_state_dict(
        self,
        model_path,
        precision=torch.float32,
        device=torch.device("cpu"),
        strict=True,
        offset=0,
    ):
        """Custom state_dict loading to enable moving module on device as they are loaded

        Args:
            model_path: Model path
            precision: same as above
            device: same as above
            strict: same as above
        """
        # bitsandbytes quantize weights when .cuda() is called
        # for huge models we need to save Ram
        # so we load the weights  module by module and transfer them to GPU for quantization
        try:
            import safetensors
        except ImportError:
            raise ImportError("run: pip install safetensors, to use safetensors")
        keyfound = {}
        shards = glob(model_path + ".*.safetensors")
        if len(shards) == 0:
            raise ValueError("No safetensors file found")
        f = []
        keys_shard = {}
        for i, shard in enumerate(shards):
            f.append(safetensors.safe_open(shard, framework="pt", device="cpu"))
            for key in f[i].keys():
                keys_shard[key] = i
        if device == torch.device("cpu"):
            offset = 0
        buf_list = []
        for buf_name, buf in self.named_buffers():
            buf_list.append(buf_name)
        for name, module in self.named_modules():
            named_buf_and_param = list(module.named_buffers()) + list(
                module.named_parameters()
            )
            for param_name, param in named_buf_and_param:
                if len(param_name.split(".")) == 1:  # only last key
                    if name + "." + param_name in keys_shard.keys():

                        ckpt_t = f[keys_shard[name + "." + param_name]].get_tensor(
                            name + "." + param_name
                        )
                        self._load_param(
                            name, module, param_name, param, buf_list, ckpt_t, offset
                        )
                        keyfound[name + "." + param_name] = True
                    elif strict and "lora" not in param_name:
                        raise ValueError(
                            "Missing key in safetensors checkpoint: %s" % name
                            + "."
                            + param_name
                        )
                    if precision == torch.int8:
                        torch.quantization.quantize_dynamic(module, inplace=True)
                    else:
                        module.to(precision)
                    module.to(device)
        for key in keys_shard.keys():
            if key not in keyfound.keys() and key not in buf_list:
                raise ValueError(
                    "Extra keys in model state_dict do not match the model config %s"
                    % key
                )


class NMTModel(BaseModel):
    """NMTModel Class
    See :class:`~onmt.models.BaseModel` for options."""

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_len, bptt=False, with_align=False):
        """An NMTModel forward the src side to the encoder.
        Then the output of encoder ``enc_out`` is forwarded to the
        decoder along with the target excluding the last token.
        The decoder state is initiliazed with:
        * enc_final_hs in the case of RNNs
        * enc_out + enc_final_hs in the case of CNNs
        * src in the case of Transformer"""

        dec_in = tgt[:, :-1, :]
        enc_out, enc_final_hs, src_len = self.encoder(src, src_len)
        if not bptt:
            self.decoder.init_state(src, enc_out, enc_final_hs)
        dec_out, attns = self.decoder(
            dec_in, enc_out, src_len=src_len, with_align=with_align
        )
        return dec_out, attns

    def update_dropout(self, dropout, attention_dropout):
        self.encoder.update_dropout(dropout, attention_dropout)
        self.decoder.update_dropout(dropout, attention_dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).

        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count"""

        enc, dec = 0, 0
        for name, param in self.named_parameters():
            if "encoder" in name:
                enc += param.nelement()
            else:
                dec += param.nelement()
        if callable(log):
            log("encoder: {}".format(enc))
            log("decoder: {}".format(dec))
            log("* number of parameters: {}".format(enc + dec))
        return enc, dec


class LanguageModel(BaseModel):
    """NMTModel Class
    Currently TransformerLMDecoder is the only LM decoder implemented

    Args:
        decoder (onmt.decoders.TransformerLMDecoder): a transformer decoder"""

    def __init__(self, encoder=None, decoder=None):
        super(LanguageModel, self).__init__(encoder, decoder)
        if encoder is not None:
            raise ValueError("LanguageModel should not be used" "with an encoder")
        self.decoder = decoder

    def forward(self, src, tgt, src_len, bptt=False, with_align=False):
        """A LanguageModel forward the src side to the decoder along
        with the source lengths vector. It is a decoder only LM (cf GPT-2)"""

        if not bptt:
            self.decoder.init_state()
        dec_out, attns = self.decoder(
            src, enc_out=None, src_len=src_len, with_align=with_align
        )
        return dec_out, attns

    def update_dropout(self, dropout, attention_dropout):
        self.decoder.update_dropout(dropout, attention_dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).

        Returns: (int, int)
            encoder side parameter count
            decoder side parameter count"""

        enc, dec = 0, 0
        for name, param in self.named_parameters():
            if "decoder" in name:
                dec += param.nelement()

        if callable(log):
            # No encoder in LM, seq2seq count formatting kept
            log("encoder: {}".format(enc))
            log("decoder: {}".format(dec))
            log("* number of parameters: {}".format(enc + dec))
        return enc, dec
