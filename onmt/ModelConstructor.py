"""
This file is for models creation, which consults options
and create each Encoder and Decoder accordingly.
"""
import torch.nn as nn

import onmt
import onmt.Models
import onmt.modules
from onmt.IO import ONMTDataset
from onmt.Models import Encoder, MeanEncoder, RNNEncoder, \
                        StdRNNDecoder, InputFeedRNNDecoder, \
                        NMTModel
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


def make_encoder(encoder_type, bidirectional, rnn_type,
                 num_layers, hidden_size, cnn_kernel_width,
                 dropout, embeddings):
    """
    Encoder dispatcher function.
    Args:
        encoder_type (string): rnn, brnn, mean, transformer, or cnn.
        bidirectional (bool): bidirectional Encoder.
        rnn_type (string): LSTM or GRU.
        num_layers (int): number of Encoder layers.
        hidden_size (int): size of hidden states of a rnn.
        cnn_kernel_width (int): size of windows in the cnn.
        dropout (float): dropout probablity.
        embeddings (Embeddings): vocab embeddings for this Encoder.
    """
    if encoder_type == "transformer":
        return TransformerEncoder(num_layers, hidden_size,
                                  dropout, embeddings)
    elif encoder_type == "cnn":
        return CNNEncoder(num_layers, hidden_size,
                          cnn_kernel_width,
                          dropout, embeddings)
    elif encoder_type == "mean":
        return MeanEncoder(num_layers, embeddings)
    else:
        # "rnn" or "brnn"
        return RNNEncoder(rnn_type, bidirectional, num_layers,
                          hidden_size, dropout, embeddings)


def make_decoder(decoder_type, rnn_type, bidirectional_encoder,
                 num_layers, hidden_size, input_feed, attn_type,
                 coverage_attn, context_gate, copy_attn,
                 cnn_kernel_width, dropout, embeddings):
    """
    Decoder dispatcher function.
    Args:
        decoder_type (string): 'rnn', 'transformer' or 'cnn'.
        rnn_type (string): 'LSTM' or 'GRU'.
        bidirectional_encoder(boo): is encoder bidirectional?
        num_layers (int): number of Decoder layers.
        hidden_size (int): size of hidden states of a rnn.
        hidden_size (int): size of hidden states of a rnn.
        input_feed (int): feed the context vector at each time step to the
                          decoder(by concating the word embeddings).
        attn_type (string): the attention type to use:
                    'dot'(dotprot), 'general'(Luong), or 'mlp'(Bahdanau).
        coverage_attn (bool): train a coverage attention layer?
        context_gate (string): type of context gate to use:
                               'source', 'target', 'both'.
        copy_attn (bool): train copy attention layer?
        cnn_kernel_width (int): size of windows in the cnn.
        dropout (float): dropout probablity.
        embeddings (Embeddings): vocab embeddings for this Decoder.
    """
    if decoder_type == "transformer":
        return TransformerDecoder(num_layers, hidden_size, attn_type,
                                  copy_attn, dropout, embeddings)
    elif decoder_type == "cnn":
        return CNNDecoder(num_layers, hidden_size, attn_type,
                          copy_attn, cnn_kernel_width,
                          dropout, embeddings)
    elif input_feed:
        return InputFeedRNNDecoder(rnn_type, bidirectional_encoder,
                                   num_layers, hidden_size,
                                   attn_type, coverage_attn, context_gate,
                                   copy_attn, dropout, embeddings)
    else:
        return StdRNNDecoder(rnn_type, bidirectional_encoder,
                             num_layers, hidden_size,
                             attn_type, coverage_attn, context_gate,
                             copy_attn, dropout, embeddings)


def make_base_model(opt, model_opt, fields, checkpoint=None):
    """
    Args:
        opt: the option in current environment.
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        checkpoint: the snapshot model.
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

        encoder = Encoder(model_opt.encoder_type, model_opt.brnn,
                          model_opt.rnn_type, model_opt.enc_layers,
                          model_opt.rnn_size, model_opt.dropout,
                          src_embeddings, model_opt.cnn_kernel_width)
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
    decoder = make_decoder(model_opt.decoder_type, model_opt.rnn_type,
                           model_opt.brnn, model_opt.dec_layers,
                           model_opt.rnn_size, model_opt.input_feed,
                           model_opt.global_attention,
                           model_opt.coverage_attn,
                           model_opt.context_gate,
                           model_opt.copy_attn,
                           model_opt.cnn_kernel_width,
                           model_opt.dropout, tgt_embeddings)

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
