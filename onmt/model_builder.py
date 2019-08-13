"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters
import onmt.modules
from onmt.encoders import str2enc, BertEncoder

from onmt.decoders import str2dec

from onmt.modules import Embeddings, CopyGenerator
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser

from onmt.models import BertPreTrainingHeads, ClassificationHead, \
                        TokenGenerationHead, TokenTaggingHead
from onmt.modules.bert_embeddings import BertEmbeddings
from collections import OrderedDict


def build_embeddings(opt, text_field, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

    pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]
    word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

    num_embs = [len(f.vocab) for _, f in text_field]
    num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    fix_word_vecs = opt.fix_word_vecs_enc if for_encoder \
        else opt.fix_word_vecs_dec

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam",
        fix_word_vecs=fix_word_vecs
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
    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):
        fields = inputters.load_old_vocab(
            vocab, opt.data_type, dynamic_dict=model_opt.copy_attn
        )
    else:
        fields = vocab

    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint,
                             opt.gpu)
    if opt.fp32:
        model.float()
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def build_base_model(model_opt, fields, gpu, checkpoint=None, gpu_id=None):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        fields (dict[str, torchtext.data.Field]):
            `Field` objects for the model.
        gpu (bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        gpu_id (int or NoneType): Which GPU to use.

    Returns:
        the NMTModel.
    """

    # Build embeddings.
    if model_opt.model_type == "text":
        src_field = fields["src"]
        src_emb = build_embeddings(model_opt, src_field)
    else:
        src_emb = None

    # Build encoder.
    encoder = build_encoder(model_opt, src_emb)

    # Build decoder.
    tgt_field = fields["tgt"]
    tgt_emb = build_embeddings(model_opt, tgt_field, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        assert src_field.base_field.vocab == tgt_field.base_field.vocab, \
            "preprocess with -share_vocab if you use share_embeddings"

        tgt_emb.word_lut.weight = src_emb.word_lut.weight

    decoder = build_decoder(model_opt, tgt_emb)

    # Build NMTModel(= encoder + decoder).
    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")
    model = onmt.models.NMTModel(encoder, decoder)

    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size,
                      len(fields["tgt"].base_field.vocab)),
            Cast(torch.float32),
            gen_func
        )
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        tgt_base_field = fields["tgt"].base_field
        vocab_size = len(tgt_base_field.vocab)
        pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
        generator = CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)

    # Load the model states from checkpoint or initialize them.
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
        # end of patch for backward compatibility

        model.load_state_dict(checkpoint['model'], strict=False)
        generator.load_state_dict(checkpoint['generator'], strict=False)
    else:
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

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec)

    model.generator = generator
    model.to(device)

    return model


def build_model(model_opt, opt, fields, checkpoint):
    logger.info('Building model...')
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    logger.info(model)
    return model


def build_bert_embeddings(opt, fields):
    token_fields_vocab = fields['tokens'].vocab
    vocab_size = len(token_fields_vocab)
    emb_size = opt.word_vec_size
    bert_emb = BertEmbeddings(vocab_size, emb_size,
                              dropout=opt.dropout[0])
    return bert_emb


def build_bert_encoder(model_opt, fields, embeddings):
    bert = BertEncoder(
        embeddings, num_layers=model_opt.layers,
        d_model=model_opt.word_vec_size, heads=model_opt.heads,
        d_ff=model_opt.transformer_ff, dropout=model_opt.dropout[0],
        max_relative_positions=model_opt.max_relative_positions)
    return bert


def build_bert_generator(model_opt, fields, bert_encoder):
    """Main part for transfer learning:
       set opt.task_type to `pretraining` if want finetuning Bert;
       set opt.task_type to `classification` if want sentence level task;
       set opt.task_type to `generation` if want token level task.
       Both all_encoder_layers and pooled_output will be feed to generator,
       pretraining task will use the two,
       while only pooled_output will be used for classification generator;
       only all_encoder_layers will be used for generation generator;
    """
    task = model_opt.task_type
    dropout = model_opt.dropout[0] if type(model_opt.dropout) is list \
        else model_opt.dropout
    if task == 'pretraining':
        generator = BertPreTrainingHeads(
            bert_encoder.d_model, bert_encoder.embeddings.vocab_size)
        if model_opt.reuse_embeddings:
            generator.mask_lm.decode.weight = \
                bert_encoder.embeddings.word_embeddings.weight
    elif task == 'generation':
        generator = TokenGenerationHead(
            bert_encoder.d_model, bert_encoder.vocab_size)
        if model_opt.reuse_embeddings:
            generator.decode.weight = \
                bert_encoder.embeddings.word_embeddings.weight
    elif task == 'classification':
        n_class = len(fields["category"].vocab.stoi) #model_opt.labels
        logger.info('Generator of classification with %s class.' % n_class)
        generator = ClassificationHead(bert_encoder.d_model, n_class, dropout)
    elif task == 'tagging':
        n_class = len(fields["token_labels"].vocab.stoi)
        logger.info('Generator of tagging with %s tag.' % n_class)
        generator = TokenTaggingHead(bert_encoder.d_model, n_class, dropout)
    return generator


def build_bert_model(model_opt, opt, fields, checkpoint=None, gpu_id=None):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        fields (dict[str, torchtext.data.Field]):
            `Field` objects for the model.
        gpu (bool): whether to use gpu.
        checkpoint: the model generated by train phase, or a resumed snapshot
                    model from a stopped training.
        gpu_id (int or NoneType): Which GPU to use.

    Returns:
        the BERT model.
    """
    logger.info('Building BERT model...')
    # Build embeddings.
    bert_emb = build_bert_embeddings(model_opt, fields)

    # Build encoder.
    bert_encoder = build_bert_encoder(model_opt, fields, bert_emb)

    gpu = use_gpu(opt)
    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")

    # Build Generator.
    generator = build_bert_generator(model_opt, fields, bert_encoder)

    # Build Bert Model(= encoder + generator).
    model = nn.Sequential(OrderedDict([
                            ('bert', bert_encoder),
                            ('generator', generator)]))

    # Load the model states from checkpoint or initialize them.
    model_init = {'bert': False, 'generator': False}
    if checkpoint is not None:
        assert 'model' in checkpoint
        if model.bert.state_dict().keys() != checkpoint['model'].keys():
            raise ValueError("Provide checkpoint don't match actual model!")
        logger.info("Load Model Parameters...")
        model.bert.load_state_dict(checkpoint['model'], strict=True)
        model_init['bert'] = True
        if model.generator.state_dict().keys() == checkpoint['generator'].keys():
            logger.info("Load generator Parameters...")
            model.generator.load_state_dict(checkpoint['generator'], strict=True)
            model_init['generator'] = True

    for sub_module, is_init in model_init.items():
        if not is_init:
            logger.info("Initialize {} Parameters...".format(sub_module))
            if model_opt.param_init_normal != 0.0:
                logger.info('Initialize weights using a normal distribution')
                normal_std = model_opt.param_init_normal
                for p in getattr(model, sub_module).parameters():
                    p.data.normal_(mean=0, std=normal_std)
            elif model_opt.param_init != 0.0:
                logger.info('Initialize weights using a uniform distribution')
                for p in getattr(model, sub_module).parameters():
                    p.data.uniform_(-model_opt.param_init,
                                    model_opt.param_init)
            elif model_opt.param_init_glorot:
                logger.info('Glorot initialization')
                for p in getattr(model, sub_module).parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
            else:
                raise AttributeError("Initialization method haven't be used!")

    model.to(device)
    logger.info(model)
    return model

def load_bert_model(opt, model_path):
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    logger.info("Checkpoint from {} Loaded.".format(model_path))
    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    # ArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint['vocab']
    fields = vocab
    model = build_bert_model(model_opt, opt, fields, checkpoint, gpu_id=opt.gpu)

    if opt.fp32:
        model.float()
    model.eval()
    model.bert.eval()
    model.generator.eval()
    return fields, model, model_opt
