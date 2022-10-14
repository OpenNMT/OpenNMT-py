#!/usr/bin/env python
"""Train models with dynamic data."""
import torch
from functools import partial

# import onmt.opts as opts
from onmt.utils.distributed import ErrorHandler, consumer, batch_producer
from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger

from onmt.models.model_saver import load_checkpoint
from onmt.train_single import main as single_main, _build_train_iter

from onmt.utils.parse import ArgumentParser
from onmt.opts import train_opts


# Set sharing strategy manually instead of default based on the OS.
torch.multiprocessing.set_sharing_strategy('file_system')


def _init_train(opt):
    """Common initilization stuff for all training process."""
    ArgumentParser.validate_prepare_opts(opt)

    if opt.train_from:
        # Load checkpoint if we resume from a previous training.
        checkpoint = load_checkpoint(ckpt_path=opt.train_from)

    else:
        checkpoint = None

    return checkpoint


def train(opt):

    init_logger(opt.log_file)

    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    set_random_seed(opt.seed, False)

    checkpoint = _init_train(opt)

    train_process = partial(
        single_main,
        checkpoint=checkpoint)

    nb_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:

        queues = []
        mp = torch.multiprocessing.get_context('spawn')
        semaphore = mp.Semaphore(opt.world_size * opt.queue_size)
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            q = mp.Queue(opt.queue_size)
            queues += [q]
            procs.append(mp.Process(target=consumer, args=(
                train_process, opt, device_id, error_queue, q, semaphore),
                daemon=True))
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        producers = []
        # This does not work if we merge with the first loop, not sure why
        for device_id in range(nb_gpu):
            # Get the iterator to generate from
            train_iter = _build_train_iter(
                opt, stride=nb_gpu, offset=device_id)
            producer = mp.Process(target=batch_producer,
                                  args=(train_iter, queues[device_id],
                                        semaphore, opt, device_id),
                                  daemon=True)
            producers.append(producer)
            producers[device_id].start()
            logger.info(" Starting producer process pid: {}  ".format(
                producers[device_id].pid))
            error_handler.add_child(producers[device_id].pid)

        for p in procs:
            p.join()
        # Once training is done, we can terminate the producers
        for p in producers:
            p.terminate()

    elif nb_gpu == 1:  # case 1 GPU only
        train_process(opt, device_id=0)
    else:   # case only CPU
        train_process(opt, device_id=-1)


def _get_parser():
    parser = ArgumentParser(description='train.py')
    train_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt, unknown = parser.parse_known_args()
    train(opt)


if __name__ == "__main__":
    main()
