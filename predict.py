#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.predictor import build_classifier, build_tagger

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.utils.bert_tokenization import BertTokenizer


def main(opt):
    logger = init_logger(opt.log_file)
    opt = ArgumentParser.validate_predict_opts(opt)
    tokenizer = BertTokenizer.from_pretrained(
        opt.bert_model, do_lower_case=opt.do_lower_case)
    data_shards = split_corpus(opt.data, opt.shard_size)
    if opt.task == 'classification':
        classifier = build_classifier(opt)
        for i, data_shard in enumerate(data_shards):
            logger.info("Classify shard %d." % i)
            data = [seq.decode("utf-8") for seq in data_shard]
            classifier.classify(
                data,
                opt.batch_size,
                tokenizer,
                delimiter=opt.delimiter,
                max_seq_len=opt.max_seq_len
            )
    if opt.task == 'tagging':
        tagger = build_tagger(opt)
        for i, data_shard in enumerate(data_shards):
            logger.info("Tagging shard %d." % i)
            data = [seq.decode("utf-8") for seq in data_shard]
            tagger.tagging(
                data,
                opt.batch_size,
                tokenizer,
                delimiter=opt.delimiter,
                max_seq_len=opt.max_seq_len
            )


def _get_parser():
    parser = ArgumentParser(description='predict.py')
    opts.config_opts(parser)
    opts.predict_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
