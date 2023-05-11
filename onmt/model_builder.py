"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import onmt.modules
from onmt.encoders import str2enc
from onmt.decoders import str2dec
from onmt.inputters.inputter import dict_to_vocabs
from onmt.modules import Embeddings, CopyGenerator
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser
from onmt.constants import DefaultTokens, ModelTask
from onmt.modules import Linear, Embedding, mark_only_lora_as_trainable


def replace_8bit_linear(model, threshold=6.0, module_to_convert=""):
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError("Install bitsandbytes to use 8bit compression")
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_8bit_linear(module, threshold, module_to_convert)

        if isinstance(module, nn.Linear) and name == module_to_convert:
            model._modules[name] = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                module.bias is not None,
                has_fp16_weights=False,
                threshold=threshold,
            )
    return model


def replace_lora_linear(model, r=2, lora_alpha=1,
                        lora_dropout=0, layer=""):
    """
    Function replacing layers with LoRa layers recursively.
    Args:
        model:
        r: rank of matrix of the Low Rank layer
        lora_alpha: cf paper
        lora_dropout: cf paper
        layer: layer name of the model to be replaced
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_lora_linear(module, r, lora_alpha,
                                lora_dropout, layer)

        if isinstance(module, nn.Linear) and name == layer:
            model._modules[name] = Linear(
                module.in_features,
                module.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=module.bias is not None)
    return model


def replace_lora_embedding(model, r=2, lora_alpha=1):
    """
    Function replacing Embeddings with LoRa ones recursively.
    Args:
        model:
        r: rank of matrix of the Low Rank layer
        lora_alpha: cf paper
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_lora_embedding(module, r, lora_alpha)
        if isinstance(module, nn.Embedding):
            model._modules[name] = Embedding(
                module.num_embeddings,
                module.embedding_dim,
                r=r,
                lora_alpha=lora_alpha,
                padding_idx=module.padding_idx,
                sparse=module.sparse)
    return model


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
        word_padding_idx = vocabs['src'][DefaultTokens.PAD]
        num_word_embeddings = len(vocabs['src'])
        if 'src_feats' in vocabs:
            feat_pad_indices = \
                [fv[DefaultTokens.PAD] for fv in vocabs['src_feats']]
            num_feat_embeddings = \
                [len(fv) for fv in vocabs['src_feats']]
        freeze_word_vecs = opt.freeze_word_vecs_enc
    else:
        emb_dim = opt.tgt_word_vec_size
        word_padding_idx = vocabs['tgt'][DefaultTokens.PAD]
        num_word_embeddings = len(vocabs['tgt'])
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
        freeze_word_vecs=freeze_word_vecs
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
    dec_type = "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed \
               else opt.decoder_type
    return str2dec[dec_type].from_opt(opt, embeddings)


def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])

    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocabs = dict_to_vocabs(checkpoint['vocab'])

    # Avoid functionality on inference
    model_opt.update_vocab = False

    model = build_base_model(model_opt, vocabs, checkpoint)

    if opt.precision == 'fp32':
        model.float()
    elif opt.precision == 'fp16':
        model.half()
    elif opt.precision == 'int8':
        if opt.gpu >= 0:
            raise ValueError(
                "Dynamic 8-bit quantization is not supported on GPU")
        torch.quantization.quantize_dynamic(model, inplace=True)

    if use_gpu(opt) and opt.gpu >= 0:
        model.to(torch.device("cuda", opt.gpu))

    model.eval()
    model.generator.eval()
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
            vocabs['src'] == vocabs['tgt']
        ), "preprocess with -share_vocab if you use share_embeddings"

    if model_opt.model_task == ModelTask.SEQ2SEQ:
        encoder, src_emb = build_encoder_with_embeddings(model_opt, vocabs)
        decoder, _ = build_decoder_with_embeddings(
            model_opt,
            vocabs,
            share_embeddings=model_opt.share_embeddings,
            src_emb=src_emb,
        )
        return onmt.models.NMTModel(encoder=encoder, decoder=decoder)
    elif model_opt.model_task == ModelTask.LANGUAGE_MODEL:
        src_emb = build_src_emb(model_opt, vocabs)
        decoder, _ = build_decoder_with_embeddings(
            model_opt, vocabs, share_embeddings=True, src_emb=src_emb
        )
        return onmt.models.LanguageModel(decoder=decoder)
    else:
        raise ValueError(f"No model defined for {model_opt.model_task} task")


def use_embeddings_from_checkpoint(vocabs, model, generator, checkpoint):
    # Update vocabulary embeddings with checkpoint embeddings
    logger.info("Updating vocabulary embeddings with checkpoint embeddings")
    # Embedding layers
    enc_emb_name = 'encoder.embeddings.make_embedding.emb_luts.0.weight'
    dec_emb_name = 'decoder.embeddings.make_embedding.emb_luts.0.weight'
    model_dict = model.state_dict()
    generator_dict = generator.state_dict()
    for side, emb_name in [('src', enc_emb_name), ('tgt', dec_emb_name)]:
        if emb_name not in checkpoint['model']:
            continue
        new_tokens = []
        ckp_vocabs = dict_to_vocabs(checkpoint['vocab'])
        for i, tok in enumerate(vocabs[side].ids_to_tokens):
            if tok in ckp_vocabs[side]:
                old_i = ckp_vocabs[side].lookup_token(tok)
                model_dict[emb_name][i] = checkpoint['model'][
                    emb_name
                ][old_i]
                if side == 'tgt':
                    generator_dict['weight'][i] = checkpoint[
                        'generator'
                    ]['weight'][old_i]
                    generator_dict['bias'][i] = checkpoint[
                        'generator'
                    ]['bias'][old_i]
            else:
                # Just for debugging purposes
                new_tokens.append(tok)
        logger.info("%s: %d new tokens" % (side, len(new_tokens)))

        # Remove old vocabulary associated embeddings
        del checkpoint['model'][emb_name]
    del checkpoint['generator']['weight'], checkpoint['generator']['bias']
    model.load_state_dict(model_dict)
    generator.load_state_dict(generator_dict)


def build_base_model(model_opt, vocabs, checkpoint=None):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        vocabs (dict[str, Vocab]):
            `Field` objects for the model.
        checkpoint: the model generated by train phase, or a resumed snapshot
                    model from a stopped training.

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

    if hasattr(model_opt, 'quant_layers') and len(model_opt.quant_layers) > 0:
        for layer in model_opt.quant_layers:
            logger.info("8bit compression of layer %s" % layer)
            model = replace_8bit_linear(model, module_to_convert=layer)

    mark_lora = False
    if hasattr(model_opt, 'lora_layers') and len(model_opt.lora_layers) > 0:
        if model_opt.freeze_encoder or model_opt.freeze_decoder:
            raise ValueError("Cannot use LoRa with Enc/Dec-oder freezing")
        for layer in model_opt.lora_layers:
            logger.info("Adding LoRa layers for %s" % layer)
            model = replace_lora_linear(model, r=model_opt.lora_rank,
                                        lora_alpha=model_opt.lora_alpha,
                                        lora_dropout=model_opt.lora_dropout,
                                        layer=layer)
        mark_lora = True
    if hasattr(model_opt, 'lora_embedding') and model_opt.lora_embedding:
        if model_opt.freeze_encoder or model_opt.freeze_decoder:
            raise ValueError("Cannot use LoRa with Enc/Dec-oder freezing")
        logger.info("Adding LoRa Embeddings")
        model = replace_lora_embedding(model, r=model_opt.lora_rank,
                                       lora_alpha=model_opt.lora_alpha)
        mark_lora = True

    if mark_lora:
        mark_only_lora_as_trainable(model, bias='lora_only')

    # Build Generator.
    if not model_opt.copy_attn:
        generator = nn.Linear(model_opt.dec_hid_size,
                              len(vocabs['tgt']))
        if model_opt.share_decoder_embeddings:
            generator.weight = model.decoder.embeddings.word_lut.weight
    else:
        vocab_size = len(vocabs['tgt'])
        pad_idx = vocabs['tgt'][DefaultTokens.PAD]
        generator = CopyGenerator(model_opt.dec_hid_size, vocab_size, pad_idx)
        if model_opt.share_decoder_embeddings:
            generator.linear.weight = model.decoder.embeddings.word_lut.weight

    # Load the model states from checkpoint or initialize them.
    if checkpoint is None or model_opt.update_vocab:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model, "encoder") and hasattr(model.encoder, "embeddings"):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec)

    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}

        if '0.weight' in checkpoint['generator']:
            checkpoint['generator']['weight'] =\
                checkpoint['generator'].pop('0.weight')
        if '0.bias' in checkpoint['generator']:
            checkpoint['generator']['bias'] =\
                checkpoint['generator'].pop('0.bias')
        # end of patch for backward compatibility

        if model_opt.update_vocab:
            # Update model embeddings with those from the checkpoint
            # after initialization
            use_embeddings_from_checkpoint(vocabs, model, generator,
                                           checkpoint)

        # when using LoRa or updating the vocab (no more embeddings in ckpt)
        # => strict=False when loading state_dict
        strict = (not model_opt.update_vocab) and (not mark_lora)
        model.load_state_dict(checkpoint['model'],
                              strict=strict)
        generator.load_state_dict(checkpoint['generator'],
                                  strict=strict)

    model.generator = generator

    return model


def build_model(model_opt, opt, vocabs, checkpoint):
    logger.info('Building model...')
    model = build_base_model(model_opt, vocabs, checkpoint)

    if model_opt.freeze_encoder:
        model.encoder.requires_grad_(False)
        model.encoder.embeddings.requires_grad_()

    if model_opt.freeze_decoder:
        model.decoder.requires_grad_(False)
        model.decoder.embeddings.requires_grad_()

    if model_opt.model_dtype == 'fp16' and \
            model_opt.apex_opt_level not in ['O0', 'O1', 'O2', 'O3'] and \
            model_opt.optim == 'fusedadam':
        model.half()   # with amp pytorch requires NOT to half the model

    if use_gpu(opt):
        model.to(torch.device("cuda"))
    logger.info(model)
    return model
