""" Main training workflow """
#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import argparse

import torch
from torch import cuda


import onmt.opts as opts
from train_multi import main as multi_main
from train_single import main as single_main


def main(opt):

    if opt.rnn_type == "SRU" and not opt.gpuid:
        raise AssertionError("Using SRU requires -gpuid set.")

    if torch.cuda.is_available() and not opt.gpuid:
        print("WARNING: You have a CUDA device, should run with -gpuid 0")

    if opt.gpuid:
        cuda.set_device(opt.gpuid[0])
        if opt.seed > 0:
            torch.cuda.manual_seed(opt.seed)

    if len(opt.gpuid) > 1:
        multi_main(opt)
    else:
        single_main(opt)



if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(PARSER)
    opts.model_opts(PARSER)
    opts.train_opts(PARSER)

    OPT = PARSER.parse_args()
    main(OPT)
