import argparse
import copy
import unittest
import onmt
import torch

from torch.autograd import Variable

# This will be redundant with #104 pull. Can simply include the parameter file

parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=500,
                    help='Word embedding sizes')
parser.add_argument('-feature_vec_size', type=int, default=100,
                    help='Feature vec sizes')

parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
parser.add_argument('-rnn_type', type=str, default='LSTM',
                    choices=['LSTM', 'GRU'],
                    help="""The gate type to use in the RNNs""")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")
parser.add_argument('-copy_attn', action="store_true",
                    help='Train copy attention layer.')
parser.add_argument('-coverage_attn', action="store_true",
                    help='Train a coverage attention layer.')
parser.add_argument('-lambda_coverage', type=float, default=1,
                    help='Lambda value for coverage.')

parser.add_argument('-encoder_layer', type=str, default='rnn',
                    help="""Type of encoder layer to use.
                    Options: [rnn|mean|transformer]""")
parser.add_argument('-decoder_layer', type=str, default='rnn',
                    help='Type of decoder layer to use. [rnn|transformer]')
parser.add_argument('-context_gate', type=str, default=None,
                    choices=['source', 'target', 'both'],
                    help="""Type of context gate to use [source|target|both].
                    Do not select for no context gate.""")
parser.add_argument('-attention_type', type=str, default='dotprod',
                    choices=['dotprod', 'mlp'],
                    help="""The attention type to use:
                    dotprot (Luong) or MLP (Bahdanau)""")

# Optimization options
parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img].")
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init).
                    Use 0 to not use initialization""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-position_encoding', action='store_true',
                    help='Use a sinusoid to mark relative words positions.')
parser.add_argument('-share_decoder_embeddings', action='store_true',
                    help='Share the word and softmax embeddings for decoder.')
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")

opt = parser.parse_known_args()[0]
print(opt)


class TestModel(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)
        self.opt = opt

    # Helper to generate a vocabulary

    def get_vocab(self):
        return onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                          onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD])

    def get_batch(self, sourceL=3, bsize=1):
        # len x batch x nfeat
        test_src = Variable(torch.ones(sourceL, bsize, 1)).long()
        test_tgt = Variable(torch.ones(sourceL, bsize)).long()
        test_length = Variable(torch.ones(1, bsize).fill_(sourceL))
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
        emb = onmt.Models.Embeddings(opt, vocab)
        test_src, _, __ = self.get_batch(sourceL=sourceL,
                                         bsize=bsize)
        if opt.decoder_layer == 'transformer':
            input = torch.cat([test_src, test_src], 0)
            res = emb(input)
            compare_to = torch.zeros(sourceL*2, bsize, opt.word_vec_size)
        else:
            res = emb(test_src)
            compare_to = torch.zeros(sourceL, bsize, opt.word_vec_size)

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
        enc = onmt.Models.Encoder(opt, vocab)

        test_src, test_tgt, test_length = self.get_batch(sourceL=sourceL,
                                                         bsize=bsize)

        hidden_t, outputs = enc(test_src, test_length)

        # Initialize vectors to compare size with
        test_hid = torch.zeros(self.opt.layers, bsize, opt.rnn_size)
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
        enc = onmt.Models.Encoder(opt, vocab)
        dec = onmt.Models.Decoder(opt, vocab)
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
                   [('decoder_layer', 'transformer')]
                   ]

for p in test_embeddings:
    _add_test(p, 'embeddings_forward')

tests_encoder = [[],
                 [('encoder_layer', 'mean')],
                 # [('encoder_layer', 'transformer'),
                 # ('word_vec_size', 16), ('rnn_size', 16)],
                 []
                 ]

for p in tests_encoder:
    _add_test(p, 'encoder_forward')

tests_ntmodel = [[('rnn_type', 'GRU')],
                 [('layers', 10)],
                 [('input_feed', 0)],
                 [('decoder_layer', 'transformer'),
                  ('encoder_layer', 'transformer'),
                  ('word_vec_size', 16),
                  ('rnn_size', 16)],
                 # [('encoder_layer', 'transformer'),
                 #  ('word_vec_size', 16),
                 #  ('rnn_size', 16)],
                 [('decoder_layer', 'transformer'),
                  ('word_vec_size', 16),
                  ('rnn_size', 16)],
                 [('coverage_attn', True)],
                 [('copy_attn', True)],
                 [('attention_type', 'mlp')],
                 [('context_gate', 'both')],
                 [('context_gate', 'target')],
                 [('context_gate', 'source')],
                 [('brnn', True),
                  ('brnn_merge', 'sum')],
                 [('brnn', True)],
                 []
                 ]

for p in tests_ntmodel:
    _add_test(p, 'ntmmodel_forward')


def suite():
    # Initialize Testsuite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModel)

    return suite


if __name__ == '__main__':
    # Run Test
    unittest.TextTestRunner(verbosity=2).run(suite())
