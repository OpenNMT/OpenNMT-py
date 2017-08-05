import argparse
import copy
import unittest
import onmt
import torch
from train_opts import add_model_arguments, add_optim_arguments
from torch.autograd import Variable

# This will be redundant with #104 pull. Can simply include the parameter file

parser = argparse.ArgumentParser(description='train.py')
add_model_arguments(parser)
add_optim_arguments(parser)

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
        emb = onmt.modules.Embeddings(opt, vocab)
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
                 #  ('decoder_layer', 'transformer'),
                 #  ('position_encoding', True),
                 #  ('word_vec_size', 16), ('rnn_size', 16)],
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
                 [('decoder_layer', 'transformer'),
                  ('encoder_layer', 'transformer'),
                  ('word_vec_size', 16),
                  ('rnn_size', 16),
                  ('position_encoding', True)],
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
