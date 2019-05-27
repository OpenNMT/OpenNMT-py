#!/usr/bin/env python
"""Train models."""
import os
import signal
import torch
import queue

import onmt.opts as opts
import onmt.utils.distributed

from itertools import cycle
from onmt.utils.logging import logger
from onmt.train_single import main as single_main
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab

def main(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)

        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        checkpoint = None
        model_opt = opt
        vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab
    train_iter = build_dataset_iter("train", fields, opt)

    nb_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:
        queues = []
        mp = torch.multiprocessing.get_context('spawn')
        semaphore = mp.Semaphore(opt.world_size)
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            q = mp.Queue(1)
            queues += [q]
            procs.append(mp.Process(target=run, args=(
                opt, device_id, error_queue, q, semaphore), daemon=True))
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        
        procs.append(mp.Process(target=batch_producer,
                                args=(train_iter, queues, semaphore,),
                     daemon=True))
        procs[-1].start()
        error_handler.add_child(procs[-1].pid)

        for p in procs:
            p.join()

    elif nb_gpu == 1:  # case 1 GPU only
        single_main(opt, 0)
    else:   # case only CPU
        single_main(opt, -1)


def batch_producer(generator_to_serve, queues, semaphore):
    generator_to_serve = iter(generator_to_serve)
    
    for b in generator_to_serve:
        v = semaphore.acquire()

        for device_id, q in enumerate(queues):
            if not q.full():
                clean = True
                if clean:
                    b.dataset = None 
                    if isinstance(b.src, tuple):
                        b.src = tuple([_.to(torch.device(device_id)) 
                                       for _ in b.src])
                    else:
                        b.src = b.src.to(torch.device(device_id))
                    b.tgt = b.tgt.to(torch.device(device_id))
                    b.indices = b.indices.to(torch.device(device_id))
                    b.alignment = b.alignment.to(torch.device(device_id)) if hasattr(b, 'alignment') else None
                    b.src_map = b.src_map.to(torch.device(device_id)) if hasattr(b, 'src_map') else None
                    
                    # hack to dodge unpicklable `dict_keys`
                    b.fields = list(b.fields)
                    q.put(b, False)
                    print("[PRODUCER] Producing batch for device: %d (tgt device: %s), queue: %s, sem: %d" % (device_id, str(b.tgt.device), str(q), v))
                
                else:
                    t = [b.src, b.tgt, b.indices]
                    t = [
                        e.to(torch.device(device_id))
                        if not isinstance(e, tuple)
                        else tuple([_.to(torch.device(device_id)) for _ in e])
                        for e in t
                    ]
                    q.put(t, False)
                    print("[PRODUCER] Producing batch for device: %d (tgt device: %s), queue: %s, sem: %d" % (device_id, str(t[1].device), str(q), v))

                # print("[PRODUCER][DONE] Producing batch for device: %d (tgt device: %s), queue: %s, sem: %d" % (device_id, str(b.tgt.device), str(q), v))
                break
        else:
            raise ValueError("Hmmm, should not happen, I mean, never.")



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


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
