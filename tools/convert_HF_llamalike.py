#!/usr/bin/env python
import torch
import argparse
import pyonmttok
from argparse import Namespace
from onmt.inputters.inputter import vocabs_to_dict
from onmt.constants import DefaultTokens
from sentencepiece import SentencePieceProcessor
import os
from transformers import AutoModelForCausalLM, AutoConfig
import huggingface_hub
from safetensors.torch import save_file


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
    opt = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        opt.model_dir,
        torch_dtype=torch.float16,
        # device_map={"": "cpu"},
        trust_remote_code=True,
    )
    checkpoint = model.state_dict()

    if opt.format == "pytorch" and opt.nshards > 1:
        raise ValueError("Saving several shards in pytorch format is not supported")

    if os.path.exists(os.path.join(opt.model_dir, "tokenizer.model")):
        tokenizer_model = os.path.join(opt.model_dir, "tokenizer.model")
    else:
        directory_path, _ = os.path.split(opt.output)
        os.makedirs(directory_path, exist_ok=True)
        try:
            tokenizer_model = huggingface_hub.hf_hub_download(
                repo_id=opt.model_dir,
                filename="tokenizer.model",
                local_dir=directory_path,
            )
        except huggingface_hub.utils.EntryNotFoundError:
            print(
                "Make sure the repo contains tokenizer.model - needed for all Llama-like models"
            )
            exit()

    config = AutoConfig.from_pretrained(opt.model_dir)
    decoder_layers = config.num_hidden_layers
    src_word_vec_size = config.hidden_size
    tgt_word_vec_size = config.hidden_size
    hidden_size = config.hidden_size
    heads = config.num_attention_heads
    vocab_size = config.vocab_size
    transformer_ff = config.intermediate_size

    if hasattr(config, "num_key_value_heads") and config.num_key_value_heads != heads:
        num_kv = config.num_key_value_heads
    else:
        num_kv = 0
    if hasattr(config, "rms_norm_eps"):
        norm_eps = config.rms_norm_eps
    else:
        norm_eps = 1e-6
    if hasattr(config, "sliding_window"):
        sliding_window = config.sliding_window
    else:
        sliding_window = 0

    if hasattr(config, "quantization_config"):
        if (
            "quant_method" in config.quantization_config.keys()
            and config.quantization_config["quant_method"] == "awq"
        ):
            if "backend" in config.quantization_config.keys():
                backend = config.quantization_config["backend"]
                if backend == "llm-awq":
                    quant_type = "llm_awq"
                elif backend == "autoawq":
                    if config.quantization_config["version"].lower() == "gemm":
                        quant_type = "aawq_gemm"
                    elif config.quantization_config["version"].lower() == "gemv":
                        quant_type = "aawq_gemv"
                    else:
                        raise ValueError("Unknown quantization config")
                else:
                    raise ValueError("Unknown backend config")
            else:
                print("Backend not specified in config, using Autoawq")
                if config.quantization_config["version"].lower() == "gemm":
                    quant_type = "aawq_gemm"
                elif config.quantization_config["version"].lower() == "gemv":
                    quant_type = "aawq_gemv"
                else:
                    raise ValueError("Unknown quantization config")
        else:
            raise ValueError("Can convert only awq models for now")
        if "bits" in config.quantization_config.keys():
            w_bit = config.quantization_config["bits"]
        else:
            w_bit = config.quantization_config["w_bit"]
        if "group_size" in config.quantization_config.keys():
            group_size = config.quantization_config["group_size"]
        else:
            group_size = config.quantization_config["q_group_size"]

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
        params = ["weight"]

    onmt_cp = {}

    for shard in range(opt.nshards):

        print("starting output shard: %d/%d" % (shard + 1, opt.nshards))
        onmt_safetensor = {}

        if shard == 0:
            onmt_safetensor[
                "decoder.embeddings.make_embedding.emb_luts.0.weight"
            ] = checkpoint["model.embed_tokens.weight"]
            onmt_safetensor["decoder.layer_norm.weight"] = checkpoint[
                "model.norm.weight"
            ]

            onmt_safetensor["generator.weight"] = checkpoint["lm_head.weight"]
            onmt_safetensor["generator.bias"] = torch.zeros(
                onmt_safetensor["generator.weight"].size(0), dtype=torch.float16
            )

        for i in range(
            -(decoder_layers // -opt.nshards) * shard,
            min(-(decoder_layers // -opt.nshards) * (shard + 1), decoder_layers),
            1,
        ):
            onmt_safetensor[
                "decoder.transformer_layers." + str(i) + ".layer_norm_1.weight"
            ] = checkpoint["model.layers." + str(i) + ".input_layernorm.weight"]

            for param in params:
                onmt_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.linear_query."
                    + param
                ] = checkpoint["model.layers." + str(i) + ".self_attn.q_proj." + param]
                onmt_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.linear_keys."
                    + param
                ] = checkpoint["model.layers." + str(i) + ".self_attn.k_proj." + param]
                onmt_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.linear_values."
                    + param
                ] = checkpoint["model.layers." + str(i) + ".self_attn.v_proj." + param]

                onmt_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.final_linear."
                    + param
                ] = checkpoint["model.layers." + str(i) + ".self_attn.o_proj." + param]

                onmt_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".feed_forward.w_1."
                    + param
                ] = checkpoint["model.layers." + str(i) + ".mlp.gate_proj." + param]

                onmt_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".feed_forward.w_2."
                    + param
                ] = checkpoint["model.layers." + str(i) + ".mlp.down_proj." + param]
                onmt_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".feed_forward.w_3."
                    + param
                ] = checkpoint["model.layers." + str(i) + ".mlp.up_proj." + param]

            onmt_safetensor[
                "decoder.transformer_layers."
                + str(i)
                + ".feed_forward.layer_norm.weight"
            ] = checkpoint[
                "model.layers." + str(i) + ".post_attention_layernorm.weight"
            ]

        if shard == 0:
            vocab_size = onmt_safetensor["generator.weight"].size(0)
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

    tokenizer = Tokenizer(model_path=tokenizer_model)
    vocabs = {}
    vocab = tokenizer.vocab
    vocab[3] = DefaultTokens.PAD
    src_vocab = pyonmttok.build_vocab_from_tokens(
        vocab, maximum_size=tokenizer.n_words, special_tokens=["<unk>", "<s>", "</s>"]
    )
    vocabs["src"] = src_vocab
    vocabs["tgt"] = src_vocab
    vocabs["data_task"] = "lm"
    vocabs["decoder_start_token"] = "<s>"

    onmt_cp["vocab"] = {}
    onmt_cp["vocab"] = vocabs_to_dict(vocabs)
    
    directory_path, _ = os.path.split(opt.output)
    os.makedirs(directory_path, exist_ok=True)
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
        layer_norm="rms",
        norm_eps=norm_eps,
        pos_ffn_activation_fn="silu",
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
        rotary_interleave=False,
        heads=heads,
        sliding_window=sliding_window,
        transformer_ff=transformer_ff,
        aan_useffn=False,
        add_qkvbias=False,
        add_ffnbias=False,
        multiquery=False,
        num_kv=num_kv,
        parallel_residual=False,
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
    )
    print("Saving the pytorch file")
    if opt.output[-3:] == ".pt":
        torch.save(onmt_cp, opt.output)
    else:
        torch.save(onmt_cp, opt.output + ".pt")
