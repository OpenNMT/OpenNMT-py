#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import configargparse
import codecs
from itertools import islice
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.opts as opts

def split_corpus(path, shard_size):
    with codecs.open(path, "r", encoding="utf-8") as f:
        while True:
            shard = list(islice(f, shard_size))
            if not shard:
                break
            yield shard

def main(opt):
    translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else [None]*opt.shard_size
    shard_pairs = zip(src_shards, tgt_shards)
    dataset_paths = []

    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        translator.translate(
	    src=src_shard,
	    tgt=tgt_shard,
	    src_dir=opt.src_dir,
	    batch_size=opt.batch_size,
	    attn_debug=opt.attn_debug
	    )

if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='translate.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
