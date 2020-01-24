#!/usr/bin/env python
"""Train models using alternate dynamic data loader."""
import os
import signal
import torch

import onmt.opts as opts
import onmt.utils.distributed

from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger
from onmt.train_single import main_dynamicdata as single_main
from onmt.utils.parse import ArgumentParser

from onmt.dynamicdata.config import read_data_config, verify_shard_config
from onmt.dynamicdata.transforms import set_train_opts
from onmt.dynamicdata.vocab import load_fields, load_transforms
from onmt.dynamicdata.iterators import build_mixer
from onmt.dynamicdata.dataset import DatasetAdaptor, build_dataset_adaptor_iter

from itertools import cycle


def train(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    set_random_seed(opt.seed, False)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        #logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        #fields = checkpoint['vocab']

    data_config = read_data_config(opt.data_config)
    verify_shard_config(data_config)
    transform_models, transforms = load_transforms(data_config)
    set_train_opts(data_config, transforms)
    fields = load_fields(data_config)
    dataset_adaptor = DatasetAdaptor(fields)

    mixer, group_epochs = build_mixer(data_config, transforms, is_train=True, bucket_size=opt.bucket_size)
    train_iter = build_dataset_adaptor_iter(mixer, dataset_adaptor, opt, is_train=True)

    valid_mixer, valid_group_epochs = build_mixer(data_config, transforms, is_train=False, bucket_size=opt.bucket_size)
    valid_iter = lambda: build_dataset_adaptor_iter(valid_mixer, dataset_adaptor, opt, is_train=False)

    nb_gpu = len(opt.gpu_ranks)

    # always using producer/consumer
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
        procs.append(mp.Process(target=run, args=(
            opt, device_id, error_queue, q, semaphore,
            valid_iter if device_id == 0 else None), daemon=True))
        procs[device_id].start()
        logger.info(" Starting process pid: %d  " % procs[device_id].pid)
        error_handler.add_child(procs[device_id].pid)
    producer = mp.Process(target=batch_producer,
                            args=(train_iter, queues, semaphore, opt,),
                            daemon=True)
    producer.start()
    error_handler.add_child(producer.pid)

    for p in procs:
        p.join()
    producer.terminate()


def batch_producer(generator_to_serve, queues, semaphore, opt):
    init_logger(opt.log_file)
    set_random_seed(opt.seed, False)
    # generator_to_serve = iter(generator_to_serve)

    def next_batch():
        new_batch = next(generator_to_serve)
        semaphore.acquire()
        return new_batch

    b = next_batch(0)

    for device_id, q in cycle(enumerate(queues)):
        b.dataset = None
        if isinstance(b.src, tuple):
            b.src = tuple([x.to(torch.device(device_id))
                           for x in b.src])
        else:
            b.src = b.src.to(torch.device(device_id))
        b.tgt = b.tgt.to(torch.device(device_id))
        b.indices = b.indices.to(torch.device(device_id))
        b.alignment = b.alignment.to(torch.device(device_id)) \
            if hasattr(b, 'alignment') else None
        b.src_map = b.src_map.to(torch.device(device_id)) \
            if hasattr(b, 'src_map') else None
        b.align = b.align.to(torch.device(device_id)) \
            if hasattr(b, 'align') else None

        # hack to dodge unpicklable `dict_keys`
        b.fields = list(b.fields)
        q.put(b)
        b = next_batch()


def run(opt, device_id, error_queue, batch_queue, semaphore):
    """ run process """
    try:
        gpu_rank = onmt.utils.distributed.multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        single_main(opt, device_id, batch_queue, semaphore)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def _get_parser():
    parser = ArgumentParser(description='train.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    train(opt)


if __name__ == "__main__":
    main()
