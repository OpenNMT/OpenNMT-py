#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse

from onmt.translate.Translator import make_translator
from onmt.Utils import get_logger

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import onmt.opts


def main(opt):
    translator = make_translator(opt, report_score=True, logger=logger)
    translator.translate(opt.src_dir, opt.src, opt.tgt,
                         opt.batch_size, opt.attn_debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = get_logger(opt.log_file)
    main(opt)
