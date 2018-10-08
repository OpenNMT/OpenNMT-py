"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters
import onmt.modules
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.encoders.image_encoder import ImageEncoder

from onmt.decoders.decoder import InputFeedRNNDecoder, StdRNNDecoder
from onmt.decoders.transformer import TransformerDecoder
from onmt.decoders.cnn_decoder import CNNDecoder

from onmt.modules import Embeddings, CopyGenerator
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger


def build_embeddings(opt, word_dict, feature_dicts, for_encoder=True):
    """
    Build an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[inputters.PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[inputters.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=opt.position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=opt.feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings,
                      sparse=opt.optim == "sparseadam")


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.encoder_type == "transformer":
        return TransformerEncoder(opt.enc_layers, opt.enc_rnn_size,
                                  opt.heads, opt.transformer_ff,
                                  opt.dropout, embeddings)
    elif opt.encoder_type == "cnn":
        return CNNEncoder(opt.enc_layers, opt.enc_rnn_size,
                          opt.cnn_kernel_width,
                          opt.dropout, embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        # "rnn" or "brnn"
        return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers,
                          opt.enc_rnn_size, opt.dropout, embeddings,
                          opt.bridge)


def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if opt.decoder_type == "transformer":
        return TransformerDecoder(opt.dec_layers, opt.dec_rnn_size,
                                  opt.heads, opt.transformer_ff,
                                  opt.global_attention, opt.copy_attn,
                                  opt.self_attn_type,
                                  opt.dropout, embeddings)
    elif opt.decoder_type == "cnn":
        return CNNDecoder(opt.dec_layers, opt.dec_rnn_size,
                          opt.global_attention, opt.copy_attn,
                          opt.cnn_kernel_width, opt.dropout,
                          embeddings)
    elif opt.input_feed:
        return InputFeedRNNDecoder(opt.rnn_type, opt.brnn,
                                   opt.dec_layers, opt.dec_rnn_size,
                                   opt.global_attention,
                                   opt.global_attention_function,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   opt.copy_attn,
                                   opt.dropout,
                                   embeddings,
                                   opt.reuse_copy_attn)
    else:
        return StdRNNDecoder(opt.rnn_type, opt.brnn,
                             opt.dec_layers, opt.dec_rnn_size,
                             opt.global_attention,
                             opt.global_attention_function,
                             opt.coverage_attn,
                             opt.context_gate,
                             opt.copy_attn,
                             opt.dropout,
                             embeddings,
                             opt.reuse_copy_attn)


def load_test_model(opt, dummy_opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    fields = inputters.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type)

    model_opt = checkpoint['opt']
    if model_opt.rnn_size != -1:
        model_opt.enc_rnn_size = model_opt.rnn_size
        model_opt.dec_rnn_size = model_opt.rnn_size
        if model_opt.model_type == 'text' and \
           model_opt.enc_rnn_size != model_opt.dec_rnn_size:
                raise AssertionError("""We do not support different encoder and
                                     decoder rnn sizes for translation now.""")
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def build_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Build encoder.
    if model_opt.model_type == "text":
        src_dict = fields["src"].vocab
        feature_dicts = inputters.collect_feature_vocabs(fields, 'src')
        src_embeddings = build_embeddings(model_opt, src_dict, feature_dicts)
        encoder = build_encoder(model_opt, src_embeddings)
    elif model_opt.model_type == "img":
        if ("image_channel_size" not in model_opt.__dict__):
            image_channel_size = 3
        else:
            image_channel_size = model_opt.image_channel_size

        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.enc_rnn_size,
                               model_opt.dropout,
                               image_channel_size)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.rnn_type,
                               model_opt.enc_layers,
                               model_opt.dec_layers,
                               model_opt.brnn,
                               model_opt.enc_rnn_size,
                               model_opt.dec_rnn_size,
                               model_opt.audio_enc_pooling,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    # Build decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = inputters.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = build_embeddings(model_opt, tgt_dict,
                                      feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = build_decoder(model_opt, tgt_embeddings)

    # Build NMTModel(= encoder + decoder).
    device = torch.device("cuda" if gpu else "cpu")
    model = onmt.models.NMTModel(encoder, decoder)

    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size, len(fields["tgt"].vocab)),
            gen_func
        )
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.dec_rnn_size,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
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
                model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator
    model.to(device)

    return model


def build_model(model_opt, opt, fields, checkpoint):
    """ Build the Model """
    logger.info('Building model...')
    model = build_base_model(model_opt, fields,
                             use_gpu(opt), checkpoint)
    logger.info(model)
    return model
