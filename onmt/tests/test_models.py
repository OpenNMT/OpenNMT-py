import copy
import unittest
import math

import torch

import onmt
import onmt.inputters
import onmt.opts
from onmt.model_builder import build_embeddings, \
    build_encoder, build_decoder
from onmt.encoders.image_encoder import ImageEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.utils.parse import ArgumentParser

parser = ArgumentParser(description='train.py')
onmt.opts.model_opts(parser)
onmt.opts.train_opts(parser)

# -data option is required, but not used in this test, so dummy.
opt = parser.parse_known_args(['-data', 'dummy'])[0]


class TestModel(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)
        self.opt = opt

    def get_field(self):
        src = onmt.inputters.get_fields("text", 0, 0)["src"]
        src.base_field.build_vocab([])
        return src

    def get_batch(self, source_l=3, bsize=1):
        # len x batch x nfeat
        test_src = torch.ones(source_l, bsize, 1).long()
        test_tgt = torch.ones(source_l, bsize, 1).long()
        test_length = torch.ones(bsize).fill_(source_l).long()
        return test_src, test_tgt, test_length

    def get_batch_image(self, tgt_l=3, bsize=1, h=15, w=17):
        # batch x c x h x w
        test_src = torch.ones(bsize, 3, h, w).float()
        test_tgt = torch.ones(tgt_l, bsize, 1).long()
        test_length = None
        return test_src, test_tgt, test_length

    def get_batch_audio(self, tgt_l=7, bsize=3, sample_rate=5500,
                        window_size=0.03, t=37):
        # batch x 1 x nfft x t
        nfft = int(math.floor((sample_rate * window_size) / 2) + 1)
        test_src = torch.ones(bsize, 1, nfft, t).float()
        test_tgt = torch.ones(tgt_l, bsize, 1).long()
        test_length = torch.ones(bsize).long().fill_(tgt_l)
        return test_src, test_tgt, test_length

    def embeddings_forward(self, opt, source_l=3, bsize=1):
        '''
        Tests if the embeddings works as expected

        args:
            opt: set of options
            source_l: Length of generated input sentence
            bsize: Batchsize of generated input
        '''
        word_field = self.get_field()
        emb = build_embeddings(opt, word_field)
        test_src, _, __ = self.get_batch(source_l=source_l, bsize=bsize)
        if opt.decoder_type == 'transformer':
            input = torch.cat([test_src, test_src], 0)
            res = emb(input)
            compare_to = torch.zeros(source_l * 2, bsize,
                                     opt.src_word_vec_size)
        else:
            res = emb(test_src)
            compare_to = torch.zeros(source_l, bsize, opt.src_word_vec_size)

        self.assertEqual(res.size(), compare_to.size())

    def encoder_forward(self, opt, source_l=3, bsize=1):
        '''
        Tests if the encoder works as expected

        args:
            opt: set of options
            source_l: Length of generated input sentence
            bsize: Batchsize of generated input
        '''
        if opt.rnn_size > 0:
            opt.enc_rnn_size = opt.rnn_size
        word_field = self.get_field()
        embeddings = build_embeddings(opt, word_field)
        enc = build_encoder(opt, embeddings)

        test_src, test_tgt, test_length = self.get_batch(source_l=source_l,
                                                         bsize=bsize)

        hidden_t, outputs, test_length = enc(test_src, test_length)

        # Initialize vectors to compare size with
        test_hid = torch.zeros(self.opt.enc_layers, bsize, opt.enc_rnn_size)
        test_out = torch.zeros(source_l, bsize, opt.dec_rnn_size)

        # Ensure correct sizes and types
        self.assertEqual(test_hid.size(),
                         hidden_t[0].size(),
                         hidden_t[1].size())
        self.assertEqual(test_out.size(), outputs.size())
        self.assertEqual(type(outputs), torch.Tensor)

    def nmtmodel_forward(self, opt, source_l=3, bsize=1):
        """
        Creates a nmtmodel with a custom opt function.
        Forwards a testbatch and checks output size.

        Args:
            opt: Namespace with options
            source_l: length of input sequence
            bsize: batchsize
        """
        if opt.rnn_size > 0:
            opt.enc_rnn_size = opt.rnn_size
            opt.dec_rnn_size = opt.rnn_size
        word_field = self.get_field()

        embeddings = build_embeddings(opt, word_field)
        enc = build_encoder(opt, embeddings)

        embeddings = build_embeddings(opt, word_field, for_encoder=False)
        dec = build_decoder(opt, embeddings)

        model = onmt.models.model.NMTModel(enc, dec)

        test_src, test_tgt, test_length = self.get_batch(source_l=source_l,
                                                         bsize=bsize)
        outputs, attn = model(test_src, test_tgt, test_length)
        outputsize = torch.zeros(source_l - 1, bsize, opt.dec_rnn_size)
        # Make sure that output has the correct size and type
        self.assertEqual(outputs.size(), outputsize.size())
        self.assertEqual(type(outputs), torch.Tensor)

    def imagemodel_forward(self, opt, tgt_l=2, bsize=1, h=15, w=17):
        """
        Creates an image-to-text nmtmodel with a custom opt function.
        Forwards a testbatch and checks output size.

        Args:
            opt: Namespace with options
            source_l: length of input sequence
            bsize: batchsize
        """
        if opt.encoder_type == 'transformer' or opt.encoder_type == 'cnn':
            return

        word_field = self.get_field()

        enc = ImageEncoder(
            opt.enc_layers, opt.brnn, opt.enc_rnn_size, opt.dropout)

        embeddings = build_embeddings(opt, word_field, for_encoder=False)
        dec = build_decoder(opt, embeddings)

        model = onmt.models.model.NMTModel(enc, dec)

        test_src, test_tgt, test_length = self.get_batch_image(
            h=h, w=w,
            bsize=bsize,
            tgt_l=tgt_l)
        outputs, attn = model(test_src, test_tgt, test_length)
        outputsize = torch.zeros(tgt_l - 1, bsize, opt.dec_rnn_size)
        # Make sure that output has the correct size and type
        self.assertEqual(outputs.size(), outputsize.size())
        self.assertEqual(type(outputs), torch.Tensor)

    def audiomodel_forward(self, opt, tgt_l=7, bsize=3, t=37):
        """
        Creates a speech-to-text nmtmodel with a custom opt function.
        Forwards a testbatch and checks output size.

        Args:
            opt: Namespace with options
            source_l: length of input sequence
            bsize: batchsize
        """
        if opt.encoder_type == 'transformer' or opt.encoder_type == 'cnn':
            return
        if opt.rnn_type == 'SRU':
            return

        word_field = self.get_field()

        enc = AudioEncoder(opt.rnn_type, opt.enc_layers, opt.dec_layers,
                           opt.brnn, opt.enc_rnn_size, opt.dec_rnn_size,
                           opt.audio_enc_pooling, opt.dropout,
                           opt.sample_rate, opt.window_size)

        embeddings = build_embeddings(opt, word_field, for_encoder=False)
        dec = build_decoder(opt, embeddings)

        model = onmt.models.model.NMTModel(enc, dec)

        test_src, test_tgt, test_length = self.get_batch_audio(
            bsize=bsize,
            sample_rate=opt.sample_rate,
            window_size=opt.window_size,
            t=t, tgt_l=tgt_l)
        outputs, attn = model(test_src, test_tgt, test_length)
        outputsize = torch.zeros(tgt_l - 1, bsize, opt.dec_rnn_size)
        # Make sure that output has the correct size and type
        self.assertEqual(outputs.size(), outputsize.size())
        self.assertEqual(type(outputs), torch.Tensor)


def _add_test(param_setting, methodname):
    """
    Adds a Test to TestModel according to settings

    Args:
        param_setting: list of tuples of (param, setting)
        methodname: name of the method that gets called
    """

    def test_method(self):
        opt = copy.deepcopy(self.opt)
        if param_setting:
            for param, setting in param_setting:
                setattr(opt, param, setting)
        ArgumentParser.update_model_opts(opt)
        getattr(self, methodname)(opt)
    if param_setting:
        name = 'test_' + methodname + "_" + "_".join(
            str(param_setting).split())
    else:
        name = 'test_' + methodname + '_standard'
    setattr(TestModel, name, test_method)
    test_method.__name__ = name


'''
TEST PARAMETERS
'''
opt.brnn = False

test_embeddings = [[],
                   [('decoder_type', 'transformer')]
                   ]

for p in test_embeddings:
    _add_test(p, 'embeddings_forward')

tests_encoder = [[],
                 [('encoder_type', 'mean')],
                 # [('encoder_type', 'transformer'),
                 # ('word_vec_size', 16), ('rnn_size', 16)],
                 []
                 ]

for p in tests_encoder:
    _add_test(p, 'encoder_forward')

tests_nmtmodel = [[('rnn_type', 'GRU')],
                  [('layers', 10)],
                  [('input_feed', 0)],
                  [('decoder_type', 'transformer'),
                   ('encoder_type', 'transformer'),
                   ('src_word_vec_size', 16),
                   ('tgt_word_vec_size', 16),
                   ('rnn_size', 16)],
                  [('decoder_type', 'transformer'),
                   ('encoder_type', 'transformer'),
                   ('src_word_vec_size', 16),
                   ('tgt_word_vec_size', 16),
                   ('rnn_size', 16),
                   ('position_encoding', True)],
                  [('coverage_attn', True)],
                  [('copy_attn', True)],
                  [('global_attention', 'mlp')],
                  [('context_gate', 'both')],
                  [('context_gate', 'target')],
                  [('context_gate', 'source')],
                  [('encoder_type', "brnn"),
                   ('brnn_merge', 'sum')],
                  [('encoder_type', "brnn")],
                  [('decoder_type', 'cnn'),
                   ('encoder_type', 'cnn')],
                  [('encoder_type', 'rnn'),
                   ('global_attention', None)],
                  [('encoder_type', 'rnn'),
                   ('global_attention', None),
                   ('copy_attn', True),
                   ('copy_attn_type', 'general')],
                  [('encoder_type', 'rnn'),
                   ('global_attention', 'mlp'),
                   ('copy_attn', True),
                   ('copy_attn_type', 'general')],
                  [],
                  ]

if onmt.models.sru.check_sru_requirement():
    #   """ Only do SRU test if requirment is safisfied. """
    # SRU doesn't support input_feed.
    tests_nmtmodel.append([('rnn_type', 'SRU'), ('input_feed', 0)])

for p in tests_nmtmodel:
    _add_test(p, 'nmtmodel_forward')

for p in tests_nmtmodel:
    _add_test(p, 'imagemodel_forward')

for p in tests_nmtmodel:
    p.append(('sample_rate', 5500))
    p.append(('window_size', 0.03))
    # when reasonable, set audio_enc_pooling to 2
    for arg, val in p:
        if arg == "layers" and int(val) > 2:
            # Need lengths >= audio_enc_pooling**n_layers.
            # That condition is unrealistic for large n_layers,
            # so leave audio_enc_pooling at 1.
            break
    else:
        p.append(('audio_enc_pooling', '2'))
    _add_test(p, 'audiomodel_forward')
