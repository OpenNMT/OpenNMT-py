#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import print_function
from __future__ import division

import argparse

import torch

import onmt.opts as opts
from train_multi import main as multi_main
from train_single import main as single_main


def main(opt):

    if opt.rnn_type == "SRU" and not opt.gpuid:
        raise AssertionError("Using SRU requires -gpuid set.")

    if torch.cuda.is_available() and not opt.gpuid:
        print("WARNING: You have a CUDA device, should run with -gpuid 0")

    if len(opt.gpuid) > 1:
        multi_main(opt)
    else:
        single_main(opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)
