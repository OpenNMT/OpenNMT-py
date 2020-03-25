#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

from onmt.dynamicdata.config import read_data_config, verify_shard_config
from onmt.dynamicdata.transforms import set_train_opts
from onmt.dynamicdata.vocab import load_fields, load_transforms
from onmt.dynamicdata.iterators import yield_debug


def _get_parser():
    parser = ArgumentParser(description='debug_dynamicdata.py')

    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('--data_config', '-data_config',
              help='Path to data config yaml file. '
                   'Turns on dynamic data loader.')
    parser.add('--transforms_from_task',
              help='Apply the same transforms as for the specified '
                   'training task.')
    parser.add('--data_type', '-data_type', default="text",
              help="Type of the source input. Options: [text|img].")
    parser.add('--src', '-src', default=None, help="Source input file")
    parser.add('--tgt', '-tgt', default=None, help="Target input file")
    parser.add('--mono', '-mono', default=None, help="Monolingual input file")
    parser.add('--src_output', '-src_output', required=True, help="Source output file")
    parser.add('--tgt_output', '-tgt_output', required=True, help="Target output file")
    parser.add('--is_valid', '-is_valid',
              help="Preprocess in validation mode (instead of train)")

    group = parser.add_argument_group('Logging')
    group.add('--verbose', '-verbose', action="store_true",
              help='Print scores and predictions for each sentence')
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--log_file_level', '-log_file_level', type=str,
              action=opts.StoreLoggingLevelAction,
              choices=opts.StoreLoggingLevelAction.CHOICES,
              default="0")

    return parser


def process(opt):
    logger = init_logger(opt.log_file)
    assert opt.data_config is not None
    if opt.mono is not None:
        assert all(x is None for x in (opt.src, opt.tgt))
        files = [opt.mono]
    else:
        files = [opt.src, opt.tgt]

    transforms_from_task = opt.transforms_from_task
    data_config = read_data_config(opt.data_config)
    verify_shard_config(data_config)
    transform_models, transforms = load_transforms(data_config)
    set_train_opts(data_config, transforms)
    fields = load_fields(data_config)
    task_transforms = transforms[transforms_from_task]
    with open(opt.src_output, 'w') as src_out, \
         open(opt.tgt_output, 'w') as tgt_out:
        for tpl in yield_debug(files, transforms_from_task, task_transforms, is_train=not opt.is_valid):
            src, tgt, idx = tpl
            print(' '.join(src), file=src_out)
            print(' '.join(tgt), file=tgt_out)


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    process(opt)


if __name__ == "__main__":
    main()
