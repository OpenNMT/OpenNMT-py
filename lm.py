from __future__ import division
from builtins import bytes

import torch
import torch.cuda as cuda
import argparse
import codecs
from onmt.LanguageModel import LMPredictor

def addone(f):
    for line in f:
        yield line
    yield None

parser = argparse.ArgumentParser(description='lm.py')
#onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src', help='Source file to compute perplexity')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the sampled sentences""")
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length for sampling.')
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-mode', type=str, default='ppl',
                    choices=['ppl', 'sample'],
                    help="""Choose whether to compute the perplexity in a text
                     or to sample""")
parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=500,
                    help='Word embedding sizes')
parser.add_argument('-rnn_type', type=str, default='LSTM',
                    choices=['LSTM', 'GRU'],
                    help="""The gate type to use in the RNNs""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')

parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")

parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")
parser.add_argument('-seed', type=int, default=-1,
                    help="""Random seed used for the experiments
                    reproducibility.""")

def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, should run with -gpus 0")

    if opt.cuda:
        cuda.set_device(opt.gpu)
        if opt.seed > 0:
            torch.cuda.manual_seed(opt.seed)

    lmPredictor = LMPredictor(opt)

    srcData = []

    print("Loading data from %s" % opt.src)
    for line in addone(codecs.open(opt.src, 'r', 'utf-8')):
        if line is not None:
            srcTokens = line.split()
            srcData += [srcTokens]

    if opt.mode == 'ppl':
        print("Computing perplexity...")
        ppl = lmPredictor.computePerplexity(srcData)
        print("DATASET %s: Computed Perplexity: %f" % (opt.src, ppl))

    elif opt.mode == 'generate':
        pass

    else:
        raise NotImplementedError("Not valid mode: %s" % opt.mode)


if __name__ == "__main__":
    main()
