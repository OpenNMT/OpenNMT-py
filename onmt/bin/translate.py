#!/usr/bin/env python
# -*- coding: utf-8 -*-
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator
from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter
from onmt.inputters.inputter import IterOnDevice
from onmt.transforms import get_transforms_cls, TransformPipe
from onmt.constants import CorpusTask
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.utils.misc import use_gpu, set_random_seed


def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    ArgumentParser._get_all_transform_translate(opt)
    ArgumentParser._validate_transforms_opts(opt)
    ArgumentParser.validate_translate_opts_dynamic(opt)
    logger = init_logger(opt.log_file)

    set_random_seed(opt.seed, use_gpu(opt))

    translator = build_translator(opt, logger=logger,
                                  report_score=True)

    transforms_cls = get_transforms_cls(opt._all_transform)

    infer_iter = build_dynamic_dataset_iter(
        opt, transforms_cls, translator.vocabs, task=CorpusTask.INFER,
        copy=translator.copy_attn)

    data_transform = [
        infer_iter.transforms[name] for name in
        opt.transforms if name in infer_iter.transforms
    ]
    transform = TransformPipe.build_from(data_transform)

    if infer_iter is not None:
        infer_iter = IterOnDevice(infer_iter, opt.gpu)

    _, _ = translator._translate(
        infer_iter,
        transform=transform,
        attn_debug=opt.attn_debug,
        align_debug=opt.align_debug
        )


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser, dynamic=True)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
