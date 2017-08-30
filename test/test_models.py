import argparse
import copy
import unittest
import onmt
import torch
import opts
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='train.py')
opts.model_opts(parser)
opts.train_opts(parser)

# pretrained word vectors
parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-fix_word_vecs_enc',
                    action='store_true',
                    help="Fix word embeddings on the encoder side.")
parser.add_argument('-fix_word_vecs_dec',
                    action='store_true',
                    help="Fix word embeddings on the encoder side.")

opt = parser.parse_known_args()[0]
print(opt)


class TestModel(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)
        self.opt = opt

    # Helper to generate a vocabulary

    def get_vocab(self):
        src = onmt.IO.ONMTDataset.get_fields()["src"]
        src.build_vocab([])
        return src.vocab

    def get_batch(self, sourceL=3, bsize=1):
        # len x batch x nfeat
        test_src = Variable(torch.ones(sourceL, bsize, 1)).long()
        test_tgt = Variable(torch.ones(sourceL, bsize)).long()
        test_length = torch.ones(bsize).fill_(sourceL)
        return test_src, test_tgt, test_length

    def embeddings_forward(self, opt, sourceL=3, bsize=1):
        '''
        Tests if the embeddings works as expected

        args:
            opt: set of options
            sourceL: Length of generated input sentence
            bsize: Batchsize of generated input
        '''
        vocab = self.get_vocab()
        emb = onmt.Models.build_embeddings(opt, vocab.stoi[onmt.IO.PAD_WORD],
                                           len(vocab), for_encoder=True)
        test_src, _, __ = self.get_batch(sourceL=sourceL,
                                         bsize=bsize)
        if opt.decoder_type == 'transformer':
            input = torch.cat([test_src, test_src], 0)
            res = emb(input)
            compare_to = torch.zeros(sourceL*2, bsize, opt.src_word_vec_size)
        else:
            res = emb(test_src)
            compare_to = torch.zeros(sourceL, bsize, opt.src_word_vec_size)

        self.assertEqual(res.size(), compare_to.size())

    def encoder_forward(self, opt, sourceL=3, bsize=1):
        '''
        Tests if the encoder works as expected

        args:
            opt: set of options
            sourceL: Length of generated input sentence
            bsize: Batchsize of generated input
        '''
        vocab = self.get_vocab()
        embeddings = onmt.Models.build_embeddings(
                                    opt, vocab.stoi[onmt.IO.PAD_WORD],
                                    len(vocab), for_encoder=True)
        enc = onmt.Models.Encoder(opt.encoder_type, opt.brnn,
                                  opt.rnn_type, opt.enc_layers,
                                  opt.rnn_size, opt.dropout, embeddings)

        test_src, test_tgt, test_length = self.get_batch(sourceL=sourceL,
                                                         bsize=bsize)

        hidden_t, outputs = enc(test_src, test_length)

        # Initialize vectors to compare size with
        test_hid = torch.zeros(self.opt.enc_layers, bsize, opt.rnn_size)
        test_out = torch.zeros(sourceL, bsize, opt.rnn_size)

        # Ensure correct sizes and types
        self.assertEqual(test_hid.size(),
                         hidden_t[0].size(),
                         hidden_t[1].size())
        self.assertEqual(test_out.size(), outputs.size())
        self.assertEqual(type(outputs), torch.autograd.Variable)
        self.assertEqual(type(outputs.data), torch.FloatTensor)

    def ntmmodel_forward(self, opt, sourceL=3, bsize=1):
        """
        Creates a ntmmodel with a custom opt function.
        Forwards a testbatch anc checks output size.

        Args:
            opt: Namespace with options
            sourceL: length of input sequence
            bsize: batchsize
        """
        vocab = self.get_vocab()
        padding_idx = vocab.stoi[onmt.IO.PAD_WORD]
        embeddings = onmt.Models.build_embeddings(opt, padding_idx, len(vocab),
                                                  for_encoder=True)
        enc = onmt.Models.Encoder(opt.encoder_type, opt.brnn,
                                  opt.rnn_type, opt.enc_layers,
                                  opt.rnn_size, opt.dropout,
                                  embeddings)
        embeddings = onmt.Models.build_embeddings(opt, padding_idx, len(vocab),
                                                  for_encoder=False)
        dec = onmt.Models.Decoder(opt, embeddings)
        model = onmt.Models.NMTModel(enc, dec)

        test_src, test_tgt, test_length = self.get_batch(sourceL=sourceL,
                                                         bsize=bsize)
        outputs, attn, _ = model(test_src,
                                 test_tgt,
                                 test_length)
        outputsize = torch.zeros(sourceL-1, bsize, opt.rnn_size)
        # Make sure that output has the correct size and type
        self.assertEqual(outputs.size(), outputsize.size())
        self.assertEqual(type(outputs), torch.autograd.Variable)
        self.assertEqual(type(outputs.data), torch.FloatTensor)


def _add_test(paramSetting, methodname):
    """
    Adds a Test to TestModel according to settings

    Args:
        paramSetting: list of tuples of (param, setting)
        methodname: name of the method that gets called
    """

    def test_method(self):
        if paramSetting:
            opt = copy.deepcopy(self.opt)
            for param, setting in paramSetting:
                setattr(opt, param, setting)
        else:
            opt = self.opt
        getattr(self, methodname)(opt)
    if paramSetting:
        name = 'test_' + methodname + "_" + "_".join(str(paramSetting).split())
    else:
        name = 'test_' + methodname + '_standard'
    setattr(TestModel, name, test_method)
    test_method.__name__ = name


'''
TEST PARAMETERS
'''

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

tests_ntmodel = [[('rnn_type', 'GRU')],
                 [('layers', 10)],
                 [('input_feed', 0)],
                 [('decoder_type', 'transformer'),
                  ('encoder_type', 'transformer'),
                  ('src_word_vec_size', 16),
                  ('tgt_word_vec_size', 16),
                  ('rnn_size', 16)],
                 # [('encoder_type', 'transformer'),
                 #  ('word_vec_size', 16),
                 #  ('rnn_size', 16)],
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
                 []
                 ]

for p in tests_ntmodel:
    _add_test(p, 'ntmmodel_forward')
