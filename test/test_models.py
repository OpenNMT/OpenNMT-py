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


class TestModelInitializing(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestModelInitializing, self).__init__(*args, **kwargs)
        self.opt = opt

    # Helper to generate a vocabulary

    def get_vocab(self):
        return onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                          onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD])

    def get_batch(self, sourceL=3, bsize=1):
        # len x batch x nfeat
        test_src = Variable(torch.ones(sourceL, bsize, 1)).long()
        test_tgt = Variable(torch.ones(sourceL, bsize)).long()
        test_length = torch.autograd.Variable(torch.ones(1, bsize))
        return test_src, test_tgt, test_length

    def test_10_embeddings_init(self):
        try:
            # Initialize Dictionary
            vocab = self.get_vocab()
            onmt.Models.Embeddings(opt, vocab)
        except:
            self.fail("Embedding Initialization Failed.")

    def test_20_encoder_init(self):
        try:
            vocab = self.get_vocab()
            onmt.Models.Encoder(opt, vocab)
        except:
            self.fail("Encoder Initialization Failed.")

    def test_30_decoder_init(self):
        try:
            vocab = self.get_vocab()
            onmt.Models.Decoder(opt, vocab)
        except:
            self.fail("Decoder Initialization Failed.")

    def test_40_nmtmodel_init(self):
        try:
            vocab = self.get_vocab()
            enc = onmt.Models.Encoder(opt, vocab)
            dec = onmt.Models.Decoder(opt, vocab)
            onmt.Models.NMTModel(enc, dec)
        except:
            self.fail("NMT model Initialization Failed.")

    def ntmmodel_forward(self, opt):
        """
        Creates a ntmmodel with a custom opt function.
        Forwards a testbatch anc checks output size.

        Args:
            opt: Namespace with options
        """
        try:
            vocab = self.get_vocab()
            enc = onmt.Models.Encoder(opt, vocab)
            dec = onmt.Models.Decoder(opt, vocab)
            model = onmt.Models.NMTModel(enc, dec)

            # Test batchsize 1
            sourceL = 3
            bsize = 1

            test_src, test_tgt, test_length = self.get_batch(sourceL=sourceL,
                                                             bsize=bsize)
            outputs, attn, _ = model(test_src,
                                     test_tgt,
                                     test_length)
            outputsize = torch.zeros(sourceL-1, bsize, opt.rnn_size)
            # Make sure that output has the correct size
            self.assertEqual(outputs.size(), outputsize.size())
        except:
            self.fail("NMT model forward failed.")

    def test_41_nmtmodel_forward(self):
        """
        Test to check whether the model forward yields the correct size
        """
        self.ntmmodel_forward(self.opt)

    def test_42_nmtmodel_forward_mlp(self):
        """
        Test to check whether the model forward yields the correct size
        with mlp attention
        """
        opt = copy.deepcopy(self.opt)
        opt.attention_type = 'mlp'
        self.ntmmodel_forward(opt)

    def test_43_nmtmodel_forward_brnn(self):
        """
        Test to check whether the model forward yields the correct size
        with bidirectional encoder
        """
        opt = copy.deepcopy(self.opt)
        opt.brnn = True
        self.ntmmodel_forward(opt)

    def test_43_1_nmtmodel_forward_brnn_sum(self):
        """
        Test to check whether the model forward yields the correct size
        with bidirectional encoder and merging by sum
        """
        opt = copy.deepcopy(self.opt)
        opt.brnn = True
        opt.brnn_merge = 'sum'
        self.ntmmodel_forward(opt)

    def test_44_nmtmodel_forward_context_source(self):
        """
        Test to check whether the model forward yields the correct size
        with source context gate
        """
        opt = copy.deepcopy(self.opt)
        opt.context_gate = 'source'
        self.ntmmodel_forward(opt)

    def test_44_1_nmtmodel_forward_context_target(self):
        """
        Test to check whether the model forward yields the correct size
        with target context gate
        """
        opt = copy.deepcopy(self.opt)
        opt.context_gate = 'target'
        self.ntmmodel_forward(opt)

    def test_44_2_nmtmodel_forward_context_both(self):
        """
        Test to check whether the model forward yields the correct size
        with both context gates
        """
        opt = copy.deepcopy(self.opt)
        opt.context_gate = 'both'
        self.ntmmodel_forward(opt)

    def test_45_nmtmodel_forward_copy(self):
        """
        Test to check whether the model forward yields the correct size
        with copy attn
        """
        opt = copy.deepcopy(self.opt)
        opt.copy_attn = True
        self.ntmmodel_forward(opt)

    def test_46_nmtmodel_forward_coverage(self):
        """
        Test to check whether the model forward yields the correct size
        with coverage attention
        """
        opt = copy.deepcopy(self.opt)
        opt.coverage_attn = True
        self.ntmmodel_forward(opt)

    # NO TRANSFORMER FOR NOW
    # def test_47_nmtmodel_forward_decoder_transformer(self):
    #     """
    #     Test to check whether the model forward yields the correct size
    #     with coverage attention
    #     """
    #     opt = copy.deepcopy(self.opt)
    #     opt.decoder_layer = 'transformer'
    #     self.ntmmodel_forward(opt)

    def test_48_nmtmodel_forward_no_input_feed(self):
        """
        Test to check whether the model forward yields the correct size
        with coverage attention
        """
        opt = copy.deepcopy(self.opt)
        opt.input_feed = 0
        self.ntmmodel_forward(opt)


def suite():
    # Initialize Testsuite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModelInitializing)

    return suite


if __name__ == '__main__':
    # Run Test
    unittest.TextTestRunner(verbosity=2).run(suite())
