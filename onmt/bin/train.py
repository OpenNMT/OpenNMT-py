#!/usr/bin/env python
"""Train models with dynamic data."""
import torch
from functools import partial
from onmt.utils.distributed import ErrorHandler, spawned_train
from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser
from onmt.opts import train_opts
from onmt.train_single import main as single_main


# Set sharing strategy manually instead of default based on the OS.
# torch.multiprocessing.set_sharing_strategy('file_system')


def train(opt):
    init_logger(opt.log_file)

    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    set_random_seed(opt.seed, False)

    train_process = partial(single_main)

    nb_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:
        mp = torch.multiprocessing.get_context("spawn")
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            procs.append(
                mp.Process(
                    target=spawned_train,
                    args=(train_process, opt, device_id, error_queue),
                    daemon=False,
                )
            )
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        for p in procs:
            p.join()

    elif nb_gpu == 1:  # case 1 GPU only
        train_process(opt, device_id=0)
    else:  # case only CPU
        train_process(opt, device_id=-1)


def _get_parser():
    parser = ArgumentParser(description="train.py")
    train_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt, unknown = parser.parse_known_args()
    train(opt)


if __name__ == "__main__":
    main()
