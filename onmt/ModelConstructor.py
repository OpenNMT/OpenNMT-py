"""
This file is for models creation, which consults options
and create each Encoder and Decoder accordingly.
"""
import torch.nn as nn

import onmt
import onmt.Models
import onmt.modules
from onmt.IO import ONMTDataset
from onmt.Models import NMTModel, MeanEncoder, RNNEncoder, \
                        StdRNNDecoder, InputFeedRNNDecoder
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator, \
                         TransformerEncoder, TransformerDecoder, \
                         CNNEncoder, CNNDecoder


def make_embeddings(opt, word_padding_idx, feats_padding_idx,
                    num_word_embeddings, for_encoder,
                    num_feat_embeddings=[]):
    """
    Make an Embeddings instance.
    Args:
        opt: command-line options.
        word_padding_idx(int): padding index for words in the embeddings.
        feats_padding_idx(int): padding index for a list of features
                                in the embeddings.
        num_word_embeddings(int): size of dictionary
                                 of embedding for words.
        for_encoder(bool): make Embeddings for Encoder or Decoder?
        num_feat_embeddings([int]): list of size of dictionary
                                    of embedding for each feature.
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size
    return Embeddings(embedding_dim,
                      opt.position_encoding,
                      opt.feat_merge,
                      opt.feat_vec_exponent,
                      opt.feat_vec_size,
                      opt.dropout,
                      word_padding_idx,
                      feats_padding_idx,
                      num_word_embeddings,
                      num_feat_embeddings)


def make_encoder(opt, embeddings):
    """
    Encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this Encoder.
    """
    if opt.encoder_type == "transformer":
        return TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.dropout, embeddings)
    elif opt.encoder_type == "cnn":
        return CNNEncoder(opt.enc_layers, opt.rnn_size,
                          opt.cnn_kernel_width,
                          opt.dropout, embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        # "rnn" or "brnn"
        return RNNEncoder(opt.rnn_type, opt.brnn, opt.dec_layers,
                          opt.rnn_size, opt.dropout, embeddings)


def make_decoder(opt, embeddings):
    """
    Decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this Decoder.
    """
    if opt.decoder_type == "transformer":
        return TransformerDecoder(opt.dec_layers, opt.rnn_size,
                                  opt.global_attention, opt.copy_attn,
                                  opt.dropout, embeddings)
    elif opt.decoder_type == "cnn":
        return CNNDecoder(opt.dec_layers, opt.rnn_size,
                          opt.global_attention, opt.copy_attn,
                          opt.cnn_kernel_width, opt.dropout,
                          embeddings)
    elif opt.input_feed:
        return InputFeedRNNDecoder(opt.rnn_type, opt.brnn,
                                   opt.dec_layers, opt.rnn_size,
                                   opt.global_attention,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   opt.copy_attn,
                                   opt.dropout,
                                   embeddings)
    else:
        return StdRNNDecoder(opt.rnn_type, opt.brnn,
                             opt.dec_layers, opt.rnn_size,
                             opt.global_attention,
                             opt.coverage_attn,
                             opt.context_gate,
                             opt.copy_attn,
                             opt.dropout,
                             embeddings)


def make_base_model(opt, model_opt, fields, checkpoint=None):
    """
    Args:
        opt: the option in current environment.
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        checkpoint: the snapshot model.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make Encoder.
    if model_opt.model_type == "text":
        src_vocab = fields["src"].vocab
        feature_dicts = ONMTDataset.collect_feature_dicts(fields)
        feats_padding_idx = [feat_dict.stoi[onmt.IO.PAD_WORD]
                             for feat_dict in feature_dicts]
        num_feat_embeddings = [len(feat_dict) for feat_dict in
                               feature_dicts]
        src_embeddings = make_embeddings(
                    model_opt, src_vocab.stoi[onmt.IO.PAD_WORD],
                    feats_padding_idx, len(src_vocab), for_encoder=True,
                    num_feat_embeddings=num_feat_embeddings)

        encoder = make_encoder(model_opt, src_embeddings)
    else:
        encoder = ImageEncoder(model_opt.layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)

    # Make Decoder.
    tgt_vocab = fields["tgt"].vocab
    # TODO: prepare for a future where tgt features are possible
    feats_padding_idx = []
    tgt_embeddings = make_embeddings(model_opt,
                                     tgt_vocab.stoi[onmt.IO.PAD_WORD],
                                     feats_padding_idx,
                                     len(tgt_vocab),
                                     for_encoder=False)
    decoder = make_decoder(model_opt, tgt_embeddings)

    # Make NMTModel(= Encoder + Decoder).
    model = NMTModel(encoder, decoder)

    # Make Generator.
    if not model_opt.copy_attn:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields["tgt"].vocab)),
            nn.LogSoftmax())
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt, fields["src"].vocab,
                                  fields["tgt"].vocab)

    # Load the modle states from checkpoint.
    if checkpoint is not None:
        print('Loading model')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])

    # Make the whole model leverage GPU if indicated to do so.
    if hasattr(opt, 'gpuid'):
        cuda = len(opt.gpuid) >= 1
    elif hasattr(opt, 'gpu'):
        cuda = opt.gpu > -1
    else:
        cuda = False

    if cuda:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()
    model.generator = generator

    return model
