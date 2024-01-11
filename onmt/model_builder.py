"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn
from torch.nn.utils import skip_init
from torch.nn.init import xavier_uniform_, zeros_, uniform_
from onmt.models.model import NMTModel, LanguageModel
from onmt.encoders import str2enc
from onmt.decoders import str2dec
from onmt.inputters.inputter import dict_to_vocabs
from onmt.modules import Embeddings, CopyGenerator
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser
from onmt.models.model_saver import load_checkpoint
from onmt.constants import DefaultTokens, ModelTask
from onmt.modules.lora import (
    replace_lora_linear,
    replace_lora_embedding,
    mark_only_lora_as_trainable,
)


def build_embeddings(opt, vocabs, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        vocab.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    feat_pad_indices = []
    num_feat_embeddings = []
    if for_encoder:
        emb_dim = opt.src_word_vec_size
        word_padding_idx = vocabs["src"][DefaultTokens.PAD]
        num_word_embeddings = len(vocabs["src"])
        if "src_feats" in vocabs:
            feat_pad_indices = [fv[DefaultTokens.PAD] for fv in vocabs["src_feats"]]
            num_feat_embeddings = [len(fv) for fv in vocabs["src_feats"]]
        freeze_word_vecs = opt.freeze_word_vecs_enc
    else:
        emb_dim = opt.tgt_word_vec_size
        word_padding_idx = vocabs["tgt"][DefaultTokens.PAD]
        num_word_embeddings = len(vocabs["tgt"])
        freeze_word_vecs = opt.freeze_word_vecs_dec

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        position_encoding_type=opt.position_encoding_type,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam",
        freeze_word_vecs=freeze_word_vecs,
    )
    return emb


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    enc_type = opt.encoder_type if opt.model_type == "text" else opt.model_type
    return str2enc[enc_type].from_opt(opt, embeddings)


def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    dec_type = (
        "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed else opt.decoder_type
    )
    return str2dec[dec_type].from_opt(opt, embeddings)


def load_test_model(opt, device_id=0, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = load_checkpoint(model_path)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])

    if hasattr(model_opt, "quant_type") and model_opt.quant_type in [
        "awq_gemm",
        "awq_gemv",
    ]:  # if the loaded model is a awq quantized one, inference config cannot overwrite this
        if (
            hasattr(opt, "quant_type")
            and opt.quant_type != ""
            and opt.quant_type != model_opt.quant_type
        ):
            raise ValueError(
                "Model is a awq quantized model, cannot overwrite with another quant method"
            )

    elif hasattr(opt, "quant_type") and opt.quant_type not in [
        "awq_gemm",
        "awq_gemv",
    ]:  # we still want to be able to load fp16/32 models with bnb 4bit to minimize ram footprint
        model_opt.quant_layers = opt.quant_layers
        model_opt.quant_type = opt.quant_type
        model_opt.lora_layers = []

    if opt.world_size > 1 and opt.parallel_mode == "tensor_parallel":
        model_opt.world_size = opt.world_size
        model_opt.parallel_mode = opt.parallel_mode
        model_opt.gpu_ranks = opt.gpu_ranks
        device = torch.device("cuda", device_id)
        offset = device_id
    else:
        if use_gpu(opt):
            if len(opt.gpu_ranks) > 0:
                device_id = opt.gpu_ranks[0]
            elif opt.gpu > -1:
                device_id = opt.gpu
            device = torch.device("cuda", device_id)
        else:
            device = torch.device("cpu")
        offset = 0

    if hasattr(opt, "self_attn_type"):
        model_opt.self_attn_type = opt.self_attn_type

    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocabs = dict_to_vocabs(checkpoint["vocab"])

    # Avoid functionality on inference
    model_opt.update_vocab = False
    model_opt.attention_dropout = (
        0.0  # required to force no dropout at inference with flash
    )

    model = build_base_model(model_opt, vocabs)

    precision = torch.float32

    if opt.precision == "fp16":
        precision = torch.float16
    elif opt.precision == "int8":
        if opt.gpu >= 0:
            raise ValueError("Dynamic 8-bit quantization is not supported on GPU")
        else:
            precision = torch.float32

    logger.info("Loading data into the model")

    if "model" in checkpoint.keys():
        # weights are in the .pt file
        model.load_state_dict(
            checkpoint,
            precision=precision,
            device=device,
            strict=True,
            offset=offset,
        )
    else:
        # weights are not in the .pt checkpoint but stored in the safetensors file
        base_name = model_path[:-3] if model_path[-3:] == ".pt" else model_path
        model.load_safe_state_dict(
            base_name,
            precision=precision,
            device=device,
            strict=True,
            offset=offset,
        )

    if opt.precision == torch.int8:
        torch.quantization.quantize_dynamic(model, dtype=torch.qint8, inplace=True)

    del checkpoint

    model.eval()
    for name, module in model.named_modules():
        if hasattr(module, "dropout_p"):
            module.dropout_p = 0.0

    return vocabs, model, model_opt


def build_src_emb(model_opt, vocabs):
    # Build embeddings.
    if model_opt.model_type == "text":
        src_emb = build_embeddings(model_opt, vocabs)
    else:
        src_emb = None
    return src_emb


def build_encoder_with_embeddings(model_opt, vocabs):
    # Build encoder.
    src_emb = build_src_emb(model_opt, vocabs)
    encoder = build_encoder(model_opt, src_emb)
    return encoder, src_emb


def build_decoder_with_embeddings(
    model_opt, vocabs, share_embeddings=False, src_emb=None
):
    # Build embeddings.
    tgt_emb = build_embeddings(model_opt, vocabs, for_encoder=False)

    if share_embeddings:
        tgt_emb.word_lut.weight = src_emb.word_lut.weight

    # Build decoder.
    decoder = build_decoder(model_opt, tgt_emb)
    return decoder, tgt_emb


def build_task_specific_model(model_opt, vocabs):
    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        assert (
            vocabs["src"] == vocabs["tgt"]
        ), "preprocess with -share_vocab if you use share_embeddings"

    if model_opt.model_task == ModelTask.SEQ2SEQ:
        encoder, src_emb = build_encoder_with_embeddings(model_opt, vocabs)
        decoder, _ = build_decoder_with_embeddings(
            model_opt,
            vocabs,
            share_embeddings=model_opt.share_embeddings,
            src_emb=src_emb,
        )
        return NMTModel(encoder=encoder, decoder=decoder)
    elif model_opt.model_task == ModelTask.LANGUAGE_MODEL:
        src_emb = build_src_emb(model_opt, vocabs)
        decoder, _ = build_decoder_with_embeddings(
            model_opt, vocabs, share_embeddings=True, src_emb=src_emb
        )
        return LanguageModel(decoder=decoder)
    else:
        raise ValueError(f"No model defined for {model_opt.model_task} task")


def use_embeddings_from_checkpoint(vocabs, model, checkpoint):
    # Update vocabulary embeddings with checkpoint embeddings
    logger.info("Updating vocabulary embeddings with checkpoint embeddings")
    # Embedding layers
    enc_emb_name = "encoder.embeddings.make_embedding.emb_luts.0.weight"
    dec_emb_name = "decoder.embeddings.make_embedding.emb_luts.0.weight"
    model_dict = {k: v for k, v in model.state_dict().items() if "generator" not in k}
    generator_dict = model.generator.state_dict()
    for side, emb_name in [("src", enc_emb_name), ("tgt", dec_emb_name)]:
        if emb_name not in checkpoint["model"]:
            continue
        new_tokens = []
        ckp_vocabs = dict_to_vocabs(checkpoint["vocab"])
        for i, tok in enumerate(vocabs[side].ids_to_tokens):
            if tok in ckp_vocabs[side]:
                old_i = ckp_vocabs[side].lookup_token(tok)
                model_dict[emb_name][i] = checkpoint["model"][emb_name][old_i]
                if side == "tgt":
                    generator_dict["weight"][i] = checkpoint["generator"]["weight"][
                        old_i
                    ]
                    generator_dict["bias"][i] = checkpoint["generator"]["bias"][old_i]
            else:
                # Just for debugging purposes
                new_tokens.append(tok)
        logger.info("%s: %d new tokens" % (side, len(new_tokens)))

        # Remove old vocabulary associated embeddings
        del checkpoint["model"][emb_name]
    del checkpoint["generator"]["weight"], checkpoint["generator"]["bias"]
    fake_ckpt = {"model": model_dict, "generator": generator_dict}
    model.load_state_dict(fake_ckpt)


def build_base_model(model_opt, vocabs):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        vocabs (dict[str, Vocab]):
            `Field` objects for the model.

    Returns:
        the NMTModel.
    """

    # for back compat when attention_dropout was not defined
    try:
        model_opt.attention_dropout
    except AttributeError:
        model_opt.attention_dropout = model_opt.dropout

    # Build Model
    model = build_task_specific_model(model_opt, vocabs)

    nonlora_to_quant = [
        layer
        for layer in getattr(model_opt, "quant_layers", [])
        if layer not in getattr(model_opt, "lora_layers", [])
    ]

    if hasattr(model_opt, "quant_layers") and len(nonlora_to_quant) > 0:
        if model_opt.quant_type in ["bnb_8bit", "bnb_FP4", "bnb_NF4"]:
            logger.info(
                "%s compression of layer %s" % (model_opt.quant_type, nonlora_to_quant)
            )
            try:
                from onmt.modules.bnb_linear import replace_bnb_linear
            except ImportError:
                raise ImportError("Install bitsandbytes to use 4/8bit compression")
            model = replace_bnb_linear(
                model, module_to_convert=nonlora_to_quant, q_type=model_opt.quant_type
            )
        elif model_opt.quant_type in ["awq_gemm", "awq_gemv"]:
            logger.info(
                "%s compression of layer %s" % (model_opt.quant_type, nonlora_to_quant)
            )
            try:
                from onmt.modules.awq_linear import replace_awq_linear
            except ImportError:
                raise ImportError("Install AutoAWQ to use awq quantized model")
            model = replace_awq_linear(
                model,
                module_to_convert=nonlora_to_quant,
                w_bit=model_opt.w_bit,
                group_size=model_opt.group_size,
                q_type=model_opt.quant_type,
            )
        else:
            logger.info("compression type %s not supported." % model_opt.quant_type)

    mark_lora = False
    if hasattr(model_opt, "lora_layers") and len(model_opt.lora_layers) > 0:
        if model_opt.freeze_encoder or model_opt.freeze_decoder:
            raise ValueError("Cannot use LoRa with Enc/Dec-oder freezing")
        for layer in model_opt.lora_layers:
            if hasattr(model_opt, "quant_layers") and layer in model_opt.quant_layers:
                quant_type = model_opt.quant_type
            else:
                quant_type = None
            logger.info("Adding LoRa layers for %s quant %s" % (layer, quant_type))
            model = replace_lora_linear(
                model,
                r=model_opt.lora_rank,
                lora_alpha=model_opt.lora_alpha,
                lora_dropout=model_opt.lora_dropout,
                layer=layer,
                quant_type=quant_type,
                use_ckpting=model_opt.use_ckpting,
            )
        mark_lora = True
    if hasattr(model_opt, "lora_embedding") and model_opt.lora_embedding:
        if model_opt.freeze_encoder or model_opt.freeze_decoder:
            raise ValueError("Cannot use LoRa with Enc/Dec-oder freezing")
        logger.info("Adding LoRa Embeddings")
        model = replace_lora_embedding(
            model, r=model_opt.lora_rank, lora_alpha=model_opt.lora_alpha
        )
        mark_lora = True

    if mark_lora:
        mark_only_lora_as_trainable(model, bias="lora_only")

    # Build Generator.
    if not model_opt.copy_attn:
        generator = skip_init(
            nn.Linear,
            in_features=model_opt.dec_hid_size,
            out_features=len(vocabs["tgt"]),
        )
        if model_opt.share_decoder_embeddings:
            generator.weight = model.decoder.embeddings.word_lut.weight
    else:
        vocab_size = len(vocabs["tgt"])
        pad_idx = vocabs["tgt"][DefaultTokens.PAD]
        generator = CopyGenerator(model_opt.dec_hid_size, vocab_size, pad_idx)
        if model_opt.share_decoder_embeddings:
            generator.linear.weight = model.decoder.embeddings.word_lut.weight

    model.generator = generator

    return model


def build_model(model_opt, opt, vocabs, checkpoint, device_id):
    logger.info("Building model...")

    model = build_base_model(model_opt, vocabs)

    # If new training initialize the model params
    # If update_vocab init also but checkpoint will overwrite old weights
    if checkpoint is None or model_opt.update_vocab:
        if model_opt.param_init != 0.0:
            for param in model.parameters():
                uniform_(param, -model_opt.param_init, model_opt.param_init)
        elif model_opt.param_init_glorot:
            for name, module in model.named_modules():
                for param_name, param in module.named_parameters():
                    if param_name == "weight" and param.dim() > 1:
                        xavier_uniform_(param)
                    elif param_name == "bias":
                        zeros_(param)
        else:
            raise ValueError("You need either param_init != 0 OR init_glorot True")

        if hasattr(model, "encoder") and hasattr(model.encoder, "embeddings"):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc
            )
        if hasattr(model.decoder, "embeddings"):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec
            )

    # ONLY for legacy fusedam with amp pytorch requires NOT to half the model
    if (
        model_opt.model_dtype == "fp16"
        and model_opt.apex_opt_level not in ["O0", "O1", "O2", "O3"]
        and model_opt.optim == "fusedadam"
    ):
        precision = torch.float16
        logger.info("Switching model to half() for FusedAdam legacy")
        logger.info("Non quantized layer compute is %s", model_opt.model_dtype)
    else:
        precision = torch.float32
        logger.info("Switching model to float32 for amp/apex_amp")
        logger.info("Non quantized layer compute is %s", model_opt.model_dtype)

    if opt.world_size > 1 and opt.parallel_mode == "tensor_parallel":
        device = torch.device("cuda")
        offset = device_id
    else:
        if use_gpu(opt):
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        offset = 0

    if checkpoint is not None:
        if model_opt.update_vocab:
            if "model" in checkpoint.keys():
                # Update model embeddings with those from the checkpoint
                # after initialization
                use_embeddings_from_checkpoint(vocabs, model, checkpoint)
                # after this checkpoint contains no embeddings
            else:
                raise ValueError(
                    "Update Vocab is not compatible with safetensors mode (yet"
                )

        # when using LoRa or updating the vocab (no more embeddings in ckpt)
        # => strict=False when loading state_dict
        strict = not model_opt.update_vocab

        if "model" in checkpoint.keys():
            # weights are in the .pt file
            model.load_state_dict(
                checkpoint,
                precision=precision,
                device=device,
                strict=strict,
                offset=offset,
            )
        else:
            # weights are not in the .pt checkpoint but stored in the safetensors file
            model_path = (
                opt.train_from[:-3] if opt.train_from[-3:] == ".pt" else opt.train_from
            )
            model.load_safe_state_dict(
                model_path,
                precision=precision,
                device=device,
                strict=strict,
                offset=offset,
            )
    else:
        model.to(precision)
        model.to(device)

    if model_opt.freeze_encoder:
        model.encoder.requires_grad_(False)
        model.encoder.embeddings.requires_grad_()

    if model_opt.freeze_decoder:
        model.decoder.requires_grad_(False)
        model.decoder.embeddings.requires_grad_()

    logger.info(model)
    return model
