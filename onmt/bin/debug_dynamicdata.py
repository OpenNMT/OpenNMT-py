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
    group.add('--data_config', '-data_config',
              help='Path to data config yaml file. '
                   'Turns on dynamic data loader.')
    group.add('--transforms_from_task',
              help='Apply the same transforms as for the specified '
                   'training task.')
    group.add('--data_type', '-data_type', default="text",
              help="Type of the source input. Options: [text|img].")
    group.add('--src', '-src', default=None, help="Source input file")
    group.add('--tgt', '-tgt', default=None, help="Target input file")
    group.add('--mono', '-mono', default=None, help="Monolingual input file")
    group.add('--src_output', '-src_output', help="Source output file")
    group.add('--tgt_output', '-tgt_output', help="Target output file")
    group.add('--is_valid', '-is_valid',
              help="Preprocess in validation mode (instead of train)")

    return parser


def process(opt):
    logger = init_logger(opt.log_file)
    assert opt.data_config is not None
    if opt.mono is not None:
        assert all(x is None) for x in (opt.src, opt.tgt)
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
    for tpl in yield_debug(files, transforms_from_task, task_transforms, is_train=not opt.is_valid):
        print(tpl)


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    process(opt)


if __name__ == "__main__":
    main()
