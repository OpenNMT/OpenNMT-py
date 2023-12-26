""" Pytorch Distributed utils
    This piece of code was heavily inspired by the equivalent of Fairseq-py
    https://github.com/pytorch/fairseq
"""
import os
import signal
import math
import pickle
import torch.distributed
from datetime import timedelta
from onmt.translate.translator import build_translator
from onmt.transforms import get_transforms_cls
from onmt.constants import CorpusTask
from onmt.utils.logging import init_logger, logger
from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter


def is_master(opt, device_id):
    return opt.gpu_ranks[device_id] == 0


def multi_init(opt, device_id):
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip=opt.master_ip, master_port=opt.master_port
    )
    dist_world_size = opt.world_size
    torch.distributed.init_process_group(
        backend=opt.gpu_backend,
        init_method=dist_init_method,
        world_size=dist_world_size,
        rank=opt.gpu_ranks[device_id],
        timeout=timedelta(seconds=opt.timeout),
    )
    gpu_rank = torch.distributed.get_rank()
    if not is_master(opt, device_id):
        logger.disabled = True

    return gpu_rank


def all_reduce_and_rescale_tensors(tensors, rescale_denom, buffer_size=104857600):
    """All-reduce and rescale tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
        buffer_size: all-reduce chunk size in bytes
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    buffer_t = (
        tensors[0].new(math.ceil(buffer_size / tensors[0].element_size())).zero_()
    )
    buffer = []

    def all_reduce_buffer():
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel()
            buffer_t[offset : offset + numel].copy_(t.view(-1))
            offset += numel

        # all-reduce and rescale
        torch.distributed.all_reduce(buffer_t[:offset], async_op=False)
        buffer_t.div_(rescale_denom)

        # copy all-reduced buffer back into tensors
        offset = 0
        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset : offset + numel])
            offset += numel

    filled = 0
    for t in tensors:
        sz = t.numel() * t.element_size()
        # print(filled, sz)
        if sz > buffer_size:
            # tensor is bigger than buffer, all-reduce and rescale directly
            torch.distributed.all_reduce(t, async_op=False)
            t.div_(rescale_denom)
        elif filled + sz > buffer_size:
            # buffer is full, all-reduce and replace buffer with grad
            all_reduce_buffer()
            buffer = [t]
            filled = sz
        else:
            # add tensor to buffer
            buffer.append(t)
            filled += sz

    if len(buffer) > 0:
        all_reduce_buffer()


def all_gather_list(data, max_size=4096):
    """Gathers arbitrary data from all nodes into a list."""
    world_size = torch.distributed.get_world_size()
    if (
        not hasattr(all_gather_list, "_in_buffer")
        or max_size != all_gather_list._in_buffer.size()
    ):
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
        all_gather_list._out_buffers = [
            torch.cuda.ByteTensor(max_size) for i in range(world_size)
        ]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError("encoded data exceeds max_size: {}".format(enc_size + 2))
    assert max_size < 255 * 256
    in_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    in_buffer[1] = enc_size % 255
    in_buffer[2 : enc_size + 2] = torch.ByteTensor(list(enc))

    torch.distributed.all_gather(out_buffers, in_buffer.cuda())

    results = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * out_buffer[0].item()) + out_buffer[1].item()

        bytes_list = bytes(out_buffer[2 : size + 2].tolist())
        result = pickle.loads(bytes_list)
        results.append(result)
    return results


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """init error handler"""
        import signal
        import threading

        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """error handler"""
        self.children_pids.append(pid)

    def error_listener(self):
        """error listener"""
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """signal handler"""
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def spawned_train(process_fn, opt, device_id, error_queue):  # noqa: E501
    """Run `process_fn` on `device_id` with data from `batch_queue`."""
    try:
        gpu_rank = multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError(
                "An error occurred in \
                  Distributed initialization"
            )
        process_fn(opt, device_id=device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback

        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


def spawned_infer(opt, device_id, error_queue, queue_instruct, queue_result):
    """Run various functions for translation in spawned process on `device_id`."""
    try:
        gpu_rank = multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError(
                "An error occurred in \
                  Distributed initialization"
            )
        torch.cuda.set_device(device_id)
        init_logger(opt.log_file)
        translator = build_translator(opt, device_id, logger=logger, report_score=True)
        transforms_cls = get_transforms_cls(opt._all_transform)
        while True:
            instruction = queue_instruct.get()
            if instruction[0] == "stop":
                break
            elif instruction[0] == "infer_list":
                src = instruction[1]
                infer_iter = build_dynamic_dataset_iter(
                    opt,
                    transforms_cls,
                    translator.vocabs,
                    task=CorpusTask.INFER,
                    src=src,
                    device_id=device_id,
                )
                scores, preds = translator._translate(
                    infer_iter, infer_iter.transforms, opt.attn_debug, opt.align_debug
                )
                queue_result.put(scores)
                queue_result.put(preds)
            elif instruction[0] == "infer_file":
                opt.src = instruction[1].src
                infer_iter = build_dynamic_dataset_iter(
                    opt,
                    transforms_cls,
                    translator.vocabs,
                    task=CorpusTask.INFER,
                    device_id=device_id,
                )
                scores, preds = translator._translate(
                    infer_iter, infer_iter.transforms, opt.attn_debug, opt.align_debug
                )
                queue_result.put(scores)
                queue_result.put(preds)

    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback

        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))
