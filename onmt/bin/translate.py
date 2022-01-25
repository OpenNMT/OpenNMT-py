#!/usr/bin/env python
# -*- coding: utf-8 -*-
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator, build_transforms
from onmt.inputters.text_dataset import InferenceDataReader

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from collections import defaultdict


def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    ArgumentParser._get_all_transform_translate(opt)
    ArgumentParser._validate_transforms_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, logger=logger, report_score=True)

    data_reader = InferenceDataReader(opt.src, opt.tgt, opt.src_feats)

    transform = build_transforms(opt, translator.fields)

    for i, (src_shard, tgt_shard, features_shard) in enumerate(data_reader.read(opt.shard_size)):
        logger.info("Translating shard %d." % i)
        translator.translate(
            src=src_shard,
            transform=transform,
            src_feats=features_shard,
            tgt=tgt_shard,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug,
            align_debug=opt.align_debug
            )


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
