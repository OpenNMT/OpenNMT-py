#!/usr/bin/env python
# -*- coding: utf-8 -*-
from onmt.inference_engine import InferenceEnginePY
from onmt.opts import translate_opts
from onmt.utils.parse import ArgumentParser
from onmt.utils.misc import use_gpu, set_random_seed
from torch.profiler import profile, record_function, ProfilerActivity
from time import time


def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    ArgumentParser._get_all_transform_translate(opt)
    ArgumentParser._validate_transforms_opts(opt)
    ArgumentParser.validate_translate_opts_dynamic(opt)

    set_random_seed(opt.seed, use_gpu(opt))

    engine = InferenceEnginePY(opt)
    _, _ = engine.infer_file()
    engine.terminate()


def _get_parser():
    parser = ArgumentParser(description="translate.py")
    translate_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()

    if opt.profile:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with record_function("Translate"):
                translate(opt)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
    else:
        init_time = time()
        translate(opt)
        print("Time w/o python interpreter load/terminate: ", time() - init_time)


if __name__ == "__main__":
    main()
