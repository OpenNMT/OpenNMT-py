#!/usr/bin/env python
import torch
import json
import argparse
import pyonmttok
import safetensors
from argparse import Namespace
from onmt.inputters.inputter import vocabs_to_dict
from onmt.constants import DefaultTokens
from sentencepiece import SentencePieceProcessor
import os
import huggingface_hub
from safetensors.torch import save_file

key_maps = {}
key_maps["LlamaForCausalLM"] = {
    "layer_prefix": "model.layers.",
    "decoder.embeddings.make_embedding.emb_luts.0.weight": "model.embed_tokens.weight",
    "decoder.layer_norm.weight": "model.norm.weight",
    "generator.weight": "lm_head.weight",
    ".self_attn.linear_query.": ".self_attn.q_proj.",
    ".self_attn.linear_keys.": ".self_attn.k_proj.",
    ".self_attn.linear_values.": ".self_attn.v_proj.",
    ".self_attn.final_linear.": ".self_attn.o_proj.",
    ".feed_forward.w_1.": ".mlp.gate_proj.",
    ".feed_forward.w_2.": ".mlp.down_proj.",
    ".feed_forward.w_3.": ".mlp.up_proj.",
    ".layer_norm_1.weight": ".input_layernorm.weight",
    ".feed_forward.layer_norm.weight": ".post_attention_layernorm.weight",
}
key_maps["MistralForCausalLM"] = key_maps["LlamaForCausalLM"]
key_maps["MixtralForCausalLM"] = {
    "layer_prefix": "model.layers.",
    "decoder.embeddings.make_embedding.emb_luts.0.weight": "model.embed_tokens.weight",
    "decoder.layer_norm.weight": "model.norm.weight",
    "generator.weight": "lm_head.weight",
    ".self_attn.linear_query.": ".self_attn.q_proj.",
    ".self_attn.linear_keys.": ".self_attn.k_proj.",
    ".self_attn.linear_values.": ".self_attn.v_proj.",
    ".self_attn.final_linear.": ".self_attn.o_proj.",
    ".layer_norm_1.weight": ".input_layernorm.weight",
    ".feed_forward.gate.weight": ".block_sparse_moe.gate.weight",
    ".feed_forward.experts.0.w_1.": ".block_sparse_moe.experts.0.w1.",
    ".feed_forward.experts.0.w_2.": ".block_sparse_moe.experts.0.w2.",
    ".feed_forward.experts.0.w_3.": ".block_sparse_moe.experts.0.w3.",
    ".feed_forward.experts.0.layer_norm.weight": ".post_attention_layernorm.weight",
    ".feed_forward.experts.1.w_1.": ".block_sparse_moe.experts.1.w1.",
    ".feed_forward.experts.1.w_2.": ".block_sparse_moe.experts.1.w2.",
    ".feed_forward.experts.1.w_3.": ".block_sparse_moe.experts.1.w3.",
    ".feed_forward.experts.1.layer_norm.weight": ".post_attention_layernorm.weight",
    ".feed_forward.experts.2.w_1.": ".block_sparse_moe.experts.2.w1.",
    ".feed_forward.experts.2.w_2.": ".block_sparse_moe.experts.2.w2.",
    ".feed_forward.experts.2.w_3.": ".block_sparse_moe.experts.2.w3.",
    ".feed_forward.experts.2.layer_norm.weight": ".post_attention_layernorm.weight",
    ".feed_forward.experts.3.w_1.": ".block_sparse_moe.experts.3.w1.",
    ".feed_forward.experts.3.w_2.": ".block_sparse_moe.experts.3.w2.",
    ".feed_forward.experts.3.w_3.": ".block_sparse_moe.experts.3.w3.",
    ".feed_forward.experts.3.layer_norm.weight": ".post_attention_layernorm.weight",
    ".feed_forward.experts.4.w_1.": ".block_sparse_moe.experts.4.w1.",
    ".feed_forward.experts.4.w_2.": ".block_sparse_moe.experts.4.w2.",
    ".feed_forward.experts.4.w_3.": ".block_sparse_moe.experts.4.w3.",
    ".feed_forward.experts.4.layer_norm.weight": ".post_attention_layernorm.weight",
    ".feed_forward.experts.5.w_1.": ".block_sparse_moe.experts.5.w1.",
    ".feed_forward.experts.5.w_2.": ".block_sparse_moe.experts.5.w2.",
    ".feed_forward.experts.5.w_3.": ".block_sparse_moe.experts.5.w3.",
    ".feed_forward.experts.5.layer_norm.weight": ".post_attention_layernorm.weight",
    ".feed_forward.experts.6.w_1.": ".block_sparse_moe.experts.6.w1.",
    ".feed_forward.experts.6.w_2.": ".block_sparse_moe.experts.6.w2.",
    ".feed_forward.experts.6.w_3.": ".block_sparse_moe.experts.6.w3.",
    ".feed_forward.experts.6.layer_norm.weight": ".post_attention_layernorm.weight",
    ".feed_forward.experts.7.w_1.": ".block_sparse_moe.experts.7.w1.",
    ".feed_forward.experts.7.w_2.": ".block_sparse_moe.experts.7.w2.",
    ".feed_forward.experts.7.w_3.": ".block_sparse_moe.experts.7.w3.",
    ".feed_forward.experts.7.layer_norm.weight": ".post_attention_layernorm.weight",
}
key_maps["PhiForCausalLM"] = {
    "layer_prefix": "model.layers.",
    "decoder.embeddings.make_embedding.emb_luts.0.weight": "model.embed_tokens.weight",
    "decoder.layer_norm.weight": "model.final_layernorm.weight",
    "decoder.layer_norm.bias": "model.final_layernorm.bias",
    "generator.weight": "lm_head.weight",
    "generator.bias": "lm_head.bias",
    ".self_attn.linear_query.": ".self_attn.q_proj.",
    ".self_attn.linear_keys.": ".self_attn.k_proj.",
    ".self_attn.linear_values.": ".self_attn.v_proj.",
    ".self_attn.final_linear.": ".self_attn.dense.",
    ".feed_forward.w_1.": ".mlp.fc1.",
    ".feed_forward.w_2.": ".mlp.fc2.",
    ".layer_norm_1.weight": (".input_layernorm.weight", ""),
    ".layer_norm_1.bias": (".input_layernorm.bias", ""),
}
ln_table = {
    "LlamaForCausalLM": "rms",
    "MistralForCausalLM": "rms",
    "MixtralForCausalLM": "rms",
    "PhiForCausalLM": "standard",
}
act_table = {
    "LlamaForCausalLM": "silu",
    "MistralForCausalLM": "silu",
    "MixtralForCausalLM": "silu",
    "PhiForCausalLM": "gelu",
}
decoder_start_table = {
    "LlamaForCausalLM": "<s>",
    "MistralForCausalLM": "<s>",
    "MixtralForCausalLM": "<s>",
    "PhiForCausalLM": "",
}


class Tokenizer:
    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
        self.vocab = [self.sp_model.id_to_piece(i) for i in range(self.n_words)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True, help="""Path to the model directory"""
    )

    parser.add_argument(
        "--output", type=str, required=True, help="""Path to the model directory"""
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pytorch",
        choices=["pytorch", "safetensors"],
        help="""Format to use 'pytorch' or 'safetensors'""",
    )
    parser.add_argument(
        "--nshards", type=int, default=1, help="""Path to the model directory"""
    )
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="""HF token""",
    )

    opt = parser.parse_args()

    if os.path.exists(opt.model_dir):
        if os.path.exists(os.path.join(opt.model_dir, "config.json")):
            config_path = os.path.join(opt.model_dir, "config.json")
        else:
            raise ValueError("You used a local directory but config.json is missing")
        if os.path.exists(os.path.join(opt.model_dir, "model.safetensors.index.json")):
            wmap_path = os.path.join(opt.model_dir, "model.safetensors.index.json")
        elif os.path.exists(
            os.path.join(opt.model_dir, "pytorch_model.bin.index.json")
        ):
            wmap_path = os.path.join(opt.model_dir, "pytorch_model.bin.index.json")
        elif os.path.exists(os.path.join(opt.model_dir, "model.safetensors")):
            wmap_path = None
            model_path = os.path.join(opt.model_dir, "model.safetensors")
        elif os.path.exists(os.path.join(opt.model_dir, "pytorch_model.bin")):
            wmap_path = None
            model_path = os.path.join(opt.model_dir, "pytorch_model.bin")
        else:
            raise ValueError(
                "Could not find any proper model configuration, please check your files"
            )
        if os.path.exists(os.path.join(opt.model_dir, "tokenizer.model")):
            tokenizer_model = os.path.join(opt.model_dir, "tokenizer.model")
        else:
            if os.path.exists(os.path.join(opt.model_dir, "tokenizer.json")):
                tokenizer_json = os.path.join(opt.model_dir, "tokenizer.json")
                tokenizer_model = None
            else:
                raise ValueError(
                    "You used a local directory but tokenizer.model",
                    " and/or tokenizer.json are missing",
                )
        if os.path.exists(os.path.join(opt.model_dir, "tokenizer_config.json")):
            tokenizer_config_json = os.path.join(opt.model_dir, "tokenizer_config.json")
        else:
            tokenizer_config_json = None
    else:
        directory_path, _ = os.path.split(opt.output)
        os.makedirs(directory_path, exist_ok=True)
        try:
            tokenizer_model = huggingface_hub.hf_hub_download(
                repo_id=opt.model_dir,
                filename="tokenizer.model",
                local_dir=directory_path,
                token=opt.token,
            )
        except huggingface_hub.utils.EntryNotFoundError:
            try:
                tokenizer_json = huggingface_hub.hf_hub_download(
                    repo_id=opt.model_dir,
                    filename="tokenizer.json",
                    local_dir=directory_path,
                    token=opt.token,
                )
                tokenizer_model = None
            except huggingface_hub.utils.EntryNotFoundError:
                raise huggingface_hub.utils.EntryNotFoundError(
                    "Make sure the repo contains tokenizer.model or tokenizer.json"
                )
        try:
            config_path = huggingface_hub.hf_hub_download(
                repo_id=opt.model_dir,
                filename="config.json",
                local_dir=directory_path,
                token=opt.token,
            )
        except huggingface_hub.utils.EntryNotFoundError:
            raise huggingface_hub.utils.EntryNotFoundError(
                "Something went wrong the repo does not contain any config.json file"
            )
        try:
            tokenizer_config_json = huggingface_hub.hf_hub_download(
                repo_id=opt.model_dir,
                filename="tokenizer_config.json",
                local_dir=directory_path,
                token=opt.token,
            )
        except huggingface_hub.utils.EntryNotFoundError:
            raise huggingface_hub.utils.EntryNotFoundError(
                "Something went wrong the repo does not contain any tokenizer_config.json file"
            )
        try:
            wmap_path = huggingface_hub.hf_hub_download(
                repo_id=opt.model_dir,
                filename="model.safetensors.index.json",
                local_dir=directory_path,
                token=opt.token,
            )
        except huggingface_hub.utils.EntryNotFoundError:
            try:
                wmap_path = huggingface_hub.hf_hub_download(
                    repo_id=opt.model_dir,
                    filename="pytorch_model.bin.index.json",
                    local_dir=directory_path,
                    token=opt.token,
                )
            except huggingface_hub.utils.EntryNotFoundError:
                try:
                    model_path = huggingface_hub.hf_hub_download(
                        repo_id=opt.model_dir,
                        filename="model.safetensors",
                        local_dir=directory_path,
                        token=opt.token,
                    )
                    wmap_path = None
                except huggingface_hub.utils.EntryNotFoundError:
                    try:
                        model_path = huggingface_hub.hf_hub_download(
                            repo_id=opt.model_dir,
                            filename="pytorch_model.bin",
                            local_dir=directory_path,
                            token=opt.token,
                        )
                        wmap_path = None
                    except huggingface_hub.utils.EntryNotFoundError:
                        raise huggingface_hub.utils.EntryNotFoundError(
                            "No valid model files found"
                        )

    with open(config_path, encoding="utf-8") as fconfig:
        config = json.load(fconfig)

    arch = config["architectures"][0]
    if "num_hidden_layers" in config.keys():
        decoder_layers = config["num_hidden_layers"]
    elif "n_layer" in config.keys():
        decoder_layers = config["n_layer"]
    else:
        raise ValueError("Can't find the number of layers in the config.json file")
    if "hidden_size" in config.keys():
        src_word_vec_size = config["hidden_size"]
        tgt_word_vec_size = config["hidden_size"]
        hidden_size = config["hidden_size"]
    elif "n_embd" in config.keys():
        src_word_vec_size = config["n_embd"]
        tgt_word_vec_size = config["n_embd"]
        hidden_size = config["n_embd"]
    else:
        raise ValueError("can't find the model hidden size in the config.json file")
    if "num_attention_heads" in config.keys():
        heads = config["num_attention_heads"]
    elif "n_head" in config.keys():
        heads = config["n_head"]
    else:
        raise ValueError("can't find the number of heads in the config.json file")
    vocab_size = config["vocab_size"]
    if "intermediate_size" in config.keys():
        transformer_ff = config["intermediate_size"]
    else:
        transformer_ff = hidden_size * 4
    pos_ffn_activation_fn = act_table[arch]
    layer_norm = ln_table[arch]

    multiquery = False
    if "multi_query" in config.keys():
        multiquery = config["multi_query"]
        num_kv = 1
    elif (
        "num_key_value_heads" in config.keys()
        and config["num_key_value_heads"] != heads
    ):
        num_kv = config["num_key_value_heads"]
    elif "num_kv_heads" in config.keys() and config["num_kv_heads"] != heads:
        num_kv = config["num_kv_heads"]
    elif "n_head_kv" in config.keys() and config["n_head_kv"] != heads:
        num_kv = config["n_head_kv"]
    else:
        num_kv = 0
    if num_kv is None:
        num_kv = 0

    shared_layer = num_kv == 1

    if "parallel_attn" in config.keys():
        parallel_residual = config["parallel_attn"]
    else:
        parallel_residual = False

    if "rms_norm_eps" in config.keys():
        norm_eps = config["rms_norm_eps"]
    elif "layer_norm_epsilon" in config.keys():
        norm_eps = config["layer_norm_epsilon"]
    elif "layer_norm_eps" in config.keys():
        norm_eps = config["layer_norm_eps"]
    else:
        norm_eps = 1e-6
    if "rope_theta" in config.keys():
        rope_theta = config["rope_theta"]
    else:
        rope_theta = 1e4
    if "rotary_dim" in config.keys():
        rotary_dim = config["rotary_dim"]
    elif "partial_rotary_factor" in config.keys():
        rotary_dim = int(config["partial_rotary_factor"] * (hidden_size // heads))
    else:
        rotary_dim = 0
    if "sliding_window" in config.keys():
        sliding_window = config["sliding_window"]
        if sliding_window is None:
            sliding_window = 4096
    else:
        sliding_window = 0
    if "num_local_experts" in config.keys():
        num_experts = config["num_local_experts"]
    else:
        num_experts = 0
    if "num_experts_per_tok" in config.keys():
        num_experts_per_tok = config["num_experts_per_tok"]
    else:
        num_experts_per_tok = 0
    if "quantization_config" in config.keys():
        if (
            "quant_method" in config["quantization_config"].keys()
            and config["quantization_config"]["quant_method"] == "awq"
        ):
            if "backend" in config["quantization_config"].keys():
                backend = config["quantization_config"]["backend"]
                if backend == "llm-awq":
                    quant_type = "awq_gemv"
                elif backend == "autoawq":
                    if config["quantization_config"]["version"].lower() == "gemm":
                        quant_type = "awq_gemm"
                    elif config["quantization_config"]["version"].lower() == "gemv":
                        quant_type = "awq_gemv"
                    else:
                        raise ValueError("Unknown quantization config")
                else:
                    raise ValueError("Unknown backend config")
            else:
                print("Backend not specified in config, using Autoawq")
                if config["quantization_config"]["version"].lower() == "gemm":
                    quant_type = "awq_gemm"
                elif config["quantization_config"]["version"].lower() == "gemv":
                    quant_type = "awq_gemv"
                else:
                    raise ValueError("Unknown quantization config")
        else:
            raise ValueError("Can convert only awq models for now")
        if "bits" in config["quantization_config"].keys():
            w_bit = config["quantization_config"]["bits"]
        else:
            w_bit = config["quantization_config"]["w_bit"]
        if "group_size" in config["quantization_config"].keys():
            group_size = config["quantization_config"]["group_size"]
        else:
            group_size = config["quantization_config"]["q_group_size"]

        quant_layers = [
            "w_1",
            "w_2",
            "w_3",
            "linear_values",
            "linear_query",
            "linear_keys",
            "final_linear",
        ]
        params = ["qweight", "qzeros", "scales"]
    else:
        quant_type = ""
        w_bit = 0
        group_size = 0
        quant_layers = []
        params = ["weight", "bias"]

    add_qkvbias = False
    add_ffnbias = False
    rotary_interleave = False
    if arch == "PhiForCausalLM":
        parallel_residual = True
        shared_layer = True
        add_qkvbias = True
        add_ffnbias = True
        rotary_interleave = False

    onmt_cp = {}

    if wmap_path:
        with open(wmap_path, encoding="utf-8") as fweights:
            wmap = json.load(fweights)

    def get_load_ckpt(dir_path, file_path):
        if os.path.exists(os.path.join(dir_path, file_path)):
            ckpt_path = os.path.join(dir_path, file_path)
        else:
            try:
                ckpt_path = huggingface_hub.hf_hub_download(
                    repo_id=opt.model_dir,
                    filename=file_path,
                    local_dir=dir_path,
                    token=opt.token,
                )
            except huggingface_hub.utils.EntryNotFoundError:
                raise huggingface_hub.utils.EntryNotFoundError(
                    "Checkpoint not found on the hub"
                )
            except PermissionError:
                ckpt_path = os.path.join(dir_path, file_path)
        if ckpt_path[-4:] == ".bin":
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        else:
            checkpoint = ckpt_path

        return checkpoint

    def get_weight(checkpoint, tensor_name):
        if isinstance(checkpoint, dict):
            if tensor_name in checkpoint.keys():
                return checkpoint[tensor_name]
            else:
                return None
        else:
            with safetensors.safe_open(checkpoint, framework="pt", device="cpu") as f:
                if tensor_name in f.keys():
                    return f.get_tensor(tensor_name)
                else:
                    return None

    for shard in range(opt.nshards):

        print("starting output shard: %d/%d" % (shard + 1, opt.nshards))
        onmt_safetensor = {}

        if shard == 0:
            targetlist = [
                "decoder.embeddings.make_embedding.emb_luts.0.weight",
                "decoder.layer_norm.weight",
                "decoder.layer_norm.bias",
                "generator.weight",
            ]
            for target in targetlist:
                if target in key_maps[arch].keys():
                    source = key_maps[arch][target]
                    if wmap_path:
                        checkpoint = get_load_ckpt(
                            os.path.split(wmap_path)[0], wmap["weight_map"][source]
                        )
                    else:
                        checkpoint = get_load_ckpt(*os.path.split(model_path))
                    w = get_weight(checkpoint, source)
                    if w is not None:
                        onmt_safetensor[target] = w

            onmt_safetensor["generator.bias"] = torch.zeros(
                onmt_safetensor["generator.weight"].size(0), dtype=torch.float16
            )

        if wmap_path:
            weightmap = wmap["weight_map"]
            ckpt_list = []
            for key in weightmap.keys():
                if (
                    key.startswith(key_maps[arch]["layer_prefix"])
                    and int(key.split(".")[2])
                    in range(
                        -(decoder_layers // -opt.nshards) * shard,
                        min(
                            -(decoder_layers // -opt.nshards) * (shard + 1),
                            decoder_layers,
                        ),
                        1,
                    )
                    and weightmap[key] not in ckpt_list
                ):
                    ckpt_list.append(weightmap[key])
                    print(weightmap[key])
        else:
            ckpt_list = [model_path]

        for ckpt in ckpt_list:
            print("Loading %s" % ckpt)
            if wmap_path:
                checkpoint = get_load_ckpt(os.path.split(wmap_path)[0], ckpt)
            else:
                checkpoint = get_load_ckpt(*os.path.split(model_path))
            for i in range(
                -(decoder_layers // -opt.nshards) * shard,
                min(-(decoder_layers // -opt.nshards) * (shard + 1), decoder_layers),
                1,
            ):
                for param in params:
                    targetlist = [
                        ".self_attn.linear_query.",
                        ".self_attn.linear_keys.",
                        ".self_attn.linear_values.",
                        ".self_attn.final_linear.",
                        ".feed_forward.w_1.",
                        ".feed_forward.w_2.",
                        ".feed_forward.w_3.",
                        ".feed_forward.experts.0.w_1.",
                        ".feed_forward.experts.0.w_2.",
                        ".feed_forward.experts.0.w_3.",
                        ".feed_forward.experts.1.w_1.",
                        ".feed_forward.experts.1.w_2.",
                        ".feed_forward.experts.1.w_3.",
                        ".feed_forward.experts.2.w_1.",
                        ".feed_forward.experts.2.w_2.",
                        ".feed_forward.experts.2.w_3.",
                        ".feed_forward.experts.3.w_1.",
                        ".feed_forward.experts.3.w_2.",
                        ".feed_forward.experts.3.w_3.",
                        ".feed_forward.experts.4.w_1.",
                        ".feed_forward.experts.4.w_2.",
                        ".feed_forward.experts.4.w_3.",
                        ".feed_forward.experts.5.w_1.",
                        ".feed_forward.experts.5.w_2.",
                        ".feed_forward.experts.5.w_3.",
                        ".feed_forward.experts.6.w_1.",
                        ".feed_forward.experts.6.w_2.",
                        ".feed_forward.experts.6.w_3.",
                        ".feed_forward.experts.7.w_1.",
                        ".feed_forward.experts.7.w_2.",
                        ".feed_forward.experts.7.w_3.",
                    ]
                    for target in targetlist:
                        if target in key_maps[arch].keys():
                            source = key_maps[arch][target]
                            if type(source) == tuple:
                                srckey = source[0]
                                srcmap = source[1]
                            else:
                                srckey = source
                            w = get_weight(
                                checkpoint,
                                key_maps[arch]["layer_prefix"]
                                + str(i)
                                + srckey
                                + param,
                            )

                            if w is not None:
                                if type(source) == tuple:
                                    w = eval("w" + srcmap)
                                onmt_safetensor[
                                    "decoder.transformer_layers."
                                    + str(i)
                                    + target
                                    + param
                                ] = w

                if shared_layer:
                    idx = 0
                else:
                    idx = 1
                for p in ["weight", "bias"]:
                    if ".layer_norm_1." + p in key_maps[arch].keys():
                        if type(key_maps[arch][".layer_norm_1." + p]) == tuple:
                            w = get_weight(
                                checkpoint,
                                key_maps[arch]["layer_prefix"]
                                + str(i)
                                + key_maps[arch][".layer_norm_1." + p][idx],
                            )
                        else:
                            w = get_weight(
                                checkpoint,
                                key_maps[arch]["layer_prefix"]
                                + str(i)
                                + key_maps[arch][".layer_norm_1." + p],
                            )

                        if w is not None:
                            onmt_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".layer_norm_1."
                                + p
                            ] = w
                    if ".layer_norm_res." + p in key_maps[arch].keys():
                        w = get_weight(
                            checkpoint,
                            key_maps[arch]["layer_prefix"]
                            + str(i)
                            + key_maps[arch][".layer_norm_res." + p],
                        )
                        if w is not None:
                            onmt_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".layer_norm_res."
                                + p
                            ] = w
                    if ".feed_forward.layer_norm." + p in key_maps[arch].keys():
                        w = get_weight(
                            checkpoint,
                            key_maps[arch]["layer_prefix"]
                            + str(i)
                            + key_maps[arch][".feed_forward.layer_norm." + p],
                        )
                        if w is not None:
                            onmt_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".feed_forward.layer_norm."
                                + p
                            ] = w

                    if ".feed_forward.gate." + p in key_maps[arch].keys():
                        w = get_weight(
                            checkpoint,
                            key_maps[arch]["layer_prefix"]
                            + str(i)
                            + key_maps[arch][".feed_forward.gate." + p],
                        )
                        if w is not None:
                            onmt_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".feed_forward.gate."
                                + p
                            ] = w

                    for j in range(num_experts):
                        if (
                            f".feed_forward.experts.{j}.layer_norm." + p
                            in key_maps[arch].keys()
                        ):
                            w = get_weight(
                                checkpoint,
                                key_maps[arch]["layer_prefix"]
                                + str(i)
                                + key_maps[arch][
                                    f".feed_forward.experts.{j}.layer_norm." + p
                                ],
                            )
                            if w is not None:
                                onmt_safetensor[
                                    "decoder.transformer_layers."
                                    + str(i)
                                    + f".feed_forward.experts.{j}.layer_norm."
                                    + p
                                ] = w

        # if shard == 0:
        #    vocab_size = onmt_safetensor["generator.weight"].size(0)
        if opt.format == "safetensors":
            print("Saving output model shard: %d" % shard)
            fileout = opt.output[:-3] if opt.output[-3:] == ".pt" else opt.output
            save_file(onmt_safetensor, fileout + ".{:02d}.safetensors".format(shard))

    if opt.format == "pytorch":
        onmt_cp["generator"] = {}
        onmt_cp["generator"]["weight"] = onmt_safetensor["generator.weight"]
        onmt_cp["generator"]["bias"] = onmt_safetensor["generator.bias"]
        del onmt_safetensor["generator.weight"]
        del onmt_safetensor["generator.bias"]
        onmt_cp["model"] = {}
        onmt_cp["model"] = onmt_safetensor

    directory_path, _ = os.path.split(opt.output)
    os.makedirs(directory_path, exist_ok=True)
    if tokenizer_config_json is not None:
        with open(tokenizer_config_json, encoding="utf-8") as f:
            data = json.load(f)
            if "add_bos_token" in data.keys():
                add_bos_token = data["add_bos_token"]
            else:
                add_bos_token = False
    else:
        add_bos_token = True
    vocabs = {}
    if tokenizer_model is not None:
        tokenizer = Tokenizer(model_path=tokenizer_model)
        vocab = tokenizer.vocab
        if "<|startoftext|>" in vocab:
            index = vocab.index("<|startoftext|>")
            vocab[index] = DefaultTokens.BOS
        if "<|endoftext|>" in vocab:
            index = vocab.index("<|endoftext|>")
            vocab[index] = DefaultTokens.EOS
        if "<0x00>" in vocab:
            index = vocab.index("<0x00>")
            vocab[index] = DefaultTokens.PAD
        src_vocab = pyonmttok.build_vocab_from_tokens(
            vocab,
            maximum_size=tokenizer.n_words,
            special_tokens=["<unk>", "<s>", "</s>"],
        )
    else:  # this section is not used for llama for now
        with open(tokenizer_json, encoding="utf-8") as f:
            data = json.load(f)
        vocab = [
            tok if tok != "Ā" else DefaultTokens.PAD for tok in data["model"]["vocab"]
        ]
        # vocab[11] = "</s>"  # Falcon only
        vocab[50256] = "</s>"  # Phi only
        src_vocab = pyonmttok.build_vocab_from_tokens(vocab)
        voc_size = len(src_vocab)
        if vocab_size > voc_size:
            for i in range(vocab_size - voc_size):
                src_vocab.add_token(DefaultTokens.VOCAB_PAD + str(i))
        with open(
            os.path.join(directory_path, "bpe.model"), "w", encoding="utf-8"
        ) as bpemodel:
            bpemodel.write("v3;false;false;false;Ġ;Ġ\n")
            for merge in data["model"]["merges"]:
                bpemodel.write(merge + "\n")

    vocabs["src"] = src_vocab
    vocabs["tgt"] = src_vocab
    vocabs["data_task"] = "lm"
    if add_bos_token:
        vocabs["decoder_start_token"] = decoder_start_table[arch]
    else:
        vocabs["decoder_start_token"] = ""
    onmt_cp["vocab"] = {}
    onmt_cp["vocab"] = vocabs_to_dict(vocabs)

    with open(
        os.path.join(directory_path, "vocab.txt"), "w", encoding="utf-8"
    ) as vocabfile:
        for tok in onmt_cp["vocab"]["src"]:
            vocabfile.write(tok + "\n")

    onmt_cp["opt"] = Namespace(
        config=None,
        save_config=None,
        data={},
        skip_empty_level="silent",
        save_data=None,
        overwrite=False,
        n_sample=0,
        dump_transforms=False,
        src_vocab=None,
        tgt_vocab=None,
        share_vocab=True,
        src_feats_vocab=None,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        vocab_size_multiple=8,
        src_words_min_frequency=0,
        tgt_words_min_frequency=0,
        decoder_start_token=vocabs["decoder_start_token"],
        src_seq_length_trunc=None,
        tgt_seq_length_trunc=None,
        both_embeddings=None,
        src_embeddings=None,
        tgt_embeddings=None,
        embeddings_type=None,
        switchout_temperature=1.0,
        tokendrop_temperature=1.0,
        tokenmask_temperature=1.0,
        reversible_tokenization=None,
        prior_tokenization=False,
        src_subword_model=None,
        tgt_subword_model=None,
        src_subword_nbest=1,
        tgt_subword_nbest=1,
        src_subword_alpha=0.0,
        tgt_subword_alpha=0.0,
        src_subword_vocab="",
        tgt_subword_vocab="",
        src_vocab_threshold=0,
        tgt_vocab_threshold=0,
        src_subword_type="none",
        tgt_subword_type="none",
        src_onmttok_kwargs="{'mode': 'none'}",
        tgt_onmttok_kwargs="{'mode': 'none'}",
        src_seq_length=512,
        tgt_seq_length=512,
        src_prefix="",
        tgt_prefix="",
        permute_sent_ratio=0.0,
        rotate_ratio=0.0,
        insert_ratio=0.0,
        random_ratio=0.0,
        mask_ratio=0.0,
        mask_length="subword",
        poisson_lambda=3.0,
        replace_length=-1,
        src_word_vec_size=src_word_vec_size,
        tgt_word_vec_size=tgt_word_vec_size,
        word_vec_size=src_word_vec_size,
        share_decoder_embeddings=False,
        share_embeddings=False,
        position_encoding=False,
        update_vocab=False,
        feat_merge="concat",
        feat_vec_size=-1,
        feat_vec_exponent=0.7,
        model_task="lm",
        model_type="text",
        model_dtype="fp16",
        encoder_type="transformer_lm",
        decoder_type="transformer_lm",
        freeze_encoder=False,
        freeze_decoder=False,
        layers=-1,
        dec_layers=decoder_layers,
        hidden_size=hidden_size,
        enc_hid_size=hidden_size,
        dec_hid_size=hidden_size,
        cnn_kernel_width=3,
        layer_norm=layer_norm,
        norm_eps=norm_eps,
        pos_ffn_activation_fn=pos_ffn_activation_fn,
        input_feed=1,
        bridge=False,
        rnn_type="LSTM",
        context_gate=None,
        bridge_extra_node=True,
        bidir_edges=True,
        state_dim=512,
        n_edge_types=2,
        n_node=2,
        n_steps=2,
        src_ggnn_size=0,
        global_attention="general",
        global_attention_function="softmax",
        self_attn_type="scaled-dot",
        max_relative_positions=-1,
        rotary_interleave=rotary_interleave,
        rotary_theta=rope_theta,
        rotary_dim=rotary_dim,
        heads=heads,
        sliding_window=sliding_window,
        transformer_ff=transformer_ff,
        aan_useffn=False,
        add_qkvbias=add_qkvbias,
        add_ffnbias=add_ffnbias,
        multiquery=multiquery,
        num_kv=num_kv,
        parallel_residual=parallel_residual,
        shared_layer_norm=shared_layer,
        lambda_align=0.0,
        alignment_layer=-3,
        alignment_heads=0,
        full_context_alignment=False,
        copy_attn=False,
        copy_attn_type="general",
        generator_function="softmax",
        copy_attn_force=False,
        reuse_copy_attn=False,
        copy_loss_by_seqlength=False,
        coverage_attn=False,
        lambda_coverage=0.0,
        lm_prior_model=None,
        lm_prior_lambda=0.0,
        lm_prior_tau=1.0,
        loss_scale=0,
        apex_opt_level="",
        data_type="text",
        save_model=None,
        save_checkpoint_steps=5000,
        keep_checkpoint=50,
        gpu_ranks=[0],
        world_size=1,
        gpu_backend="nccl",
        gpu_verbose_level=0,
        master_ip="localhost",
        master_port=10000,
        seed=1234,
        param_init=0.0,
        param_init_glorot=True,
        train_from=None,
        reset_optim="none",
        pre_word_vecs_enc=None,
        pre_word_vecs_dec=None,
        freeze_word_vecs_enc=False,
        freeze_word_vecs_dec=False,
        num_workers=2,
        batch_size=896,
        batch_size_multiple=1,
        batch_type="tokens",
        normalization="tokens",
        accum_count=[32],
        accum_steps=[0],
        valid_steps=400,
        valid_batch_size=256,
        train_steps=4000,
        single_pass=False,
        early_stopping=0,
        early_stopping_criteria=None,
        optim="fusedadam",
        adagrad_accumulator_init=0,
        max_grad_norm=0.0,
        dropout=[0.0],
        attention_dropout=[0.0],
        dropout_steps=[0],
        truncated_decoder=0,
        adam_beta1=0.9,
        adam_beta2=0.998,
        label_smoothing=0.0,
        average_decay=0,
        average_every=1,
        learning_rate=0.00002,
        learning_rate_decay=0.5,
        start_decay_steps=50000,
        decay_steps=10000,
        decay_method="none",
        warmup_steps=4000,
        log_file="",
        log_file_level="0",
        verbose=False,
        train_eval_steps=200,
        train_metrics=[],
        valid_metrics=[],
        scoring_debug=False,
        dump_preds=None,
        report_every=100,
        exp_host="",
        exp="",
        tensorboard=False,
        tensorboard_log_dir="runs/onmt",
        bucket_size=262144,
        bucket_size_init=-1,
        bucket_size_increment=0,
        prefetch_factor=400,
        brnn=False,
        data_task="lm",
        _all_transform={"filtertoolong"},
        quant_layers=quant_layers,
        quant_type=quant_type,
        w_bit=w_bit,
        group_size=group_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    print("Saving the pytorch file")
    if opt.output[-3:] == ".pt":
        torch.save(onmt_cp, opt.output)
    else:
        torch.save(onmt_cp, opt.output + ".pt")
