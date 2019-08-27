#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        # max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    # max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt[0]) + 1)
    src_elements = count * max_src_in_batch
    # tgt_elements = count * max_tgt_in_batch
    return src_elements

def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else repeat(None)
    shard_pairs = zip(src_shards, tgt_shards)

    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            batch_size_fn=max_tok_len if opt.batch_type == "tokens" else None,
            attn_debug=opt.attn_debug
            )


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
