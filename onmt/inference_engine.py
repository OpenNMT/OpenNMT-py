import torch
from onmt.utils.distributed import ErrorHandler, spawned_infer
from onmt.translate.translator import build_translator
from onmt.transforms import get_transforms_cls
from onmt.constants import CorpusTask
from onmt.utils.logging import logger
from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter
from onmt.inputters.inputter import IterOnDevice


class InferenceEngine(object):
    """Wrapper Class to run Inference in mulitpocessing with partitioned models.

    Args:
        opt: inference options
    """

    def __init__(self, opt):
        self.opt = opt

        if opt.world_size > 1:

            mp = torch.multiprocessing.get_context("spawn")
            # Create a thread to listen for errors in the child processes.
            self.error_queue = mp.SimpleQueue()
            self.error_handler = ErrorHandler(self.error_queue)
            self.queue_instruct = []
            self.queue_result = []
            self.procs = []

            print("world_size: ", opt.world_size)
            print("gpu_ranks: ", opt.gpu_ranks)
            print("opt.gpu: ", opt.gpu)

            for device_id in range(opt.world_size):
                self.queue_instruct.append(mp.Queue())
                self.queue_result.append(mp.Queue())
                self.procs.append(
                    mp.Process(
                        target=spawned_infer,
                        args=(
                            opt,
                            device_id,
                            self.error_queue,
                            self.queue_instruct[device_id],
                            self.queue_result[device_id],
                        ),
                        daemon=False,
                    )
                )
                self.procs[device_id].start()
                print(" Starting process pid: %d  " % self.procs[device_id].pid)
                self.error_handler.add_child(self.procs[device_id].pid)
        else:
            self.device_id = 0
            self.translator = build_translator(
                opt, self.device_id, logger=logger, report_score=True
            )
            self.transforms_cls = get_transforms_cls(opt._all_transform)

    def infer_file(self):
        """File inference. Source file must be the opt.src argument"""
        if self.opt.world_size > 1:
            for device_id in range(self.opt.world_size):
                self.queue_instruct[device_id].put(("infer_file", self.opt))
            scores, preds = [], []
            for device_id in range(self.opt.world_size):
                scores.append(self.queue_result[device_id].get())
                preds.append(self.queue_result[device_id].get())
            return scores[0], preds[0]
        else:
            infer_iter = build_dynamic_dataset_iter(
                self.opt,
                self.transforms_cls,
                self.translator.vocabs,
                task=CorpusTask.INFER,
            )
            infer_iter = IterOnDevice(infer_iter, self.device_id)
            scores, preds = self.translator._translate(
                infer_iter,
                infer_iter.transform,
                self.opt.attn_debug,
                self.opt.align_debug,
            )
            return scores, preds

    def infer_list(self, src):
        """List of strings inference `src`"""
        if self.opt.world_size > 1:
            for device_id in range(self.opt.world_size):
                self.queue_instruct[device_id].put(("infer_list", src))
            scores, preds = [], []
            for device_id in range(self.opt.world_size):
                scores.append(self.queue_result[device_id].get())
                preds.append(self.queue_result[device_id].get())
            return scores[0], preds[0]
        else:
            infer_iter = build_dynamic_dataset_iter(
                self.opt,
                self.transforms_cls,
                self.translator.vocabs,
                task=CorpusTask.INFER,
                src=src,
            )
            infer_iter = IterOnDevice(infer_iter, self.device_id)
            scores, preds = self.translator._translate(
                infer_iter,
                infer_iter.transform,
                self.opt.attn_debug,
                self.opt.align_debug,
            )
            return scores, preds

    def terminate(self):
        if self.opt.world_size > 1:
            for device_id in range(self.opt.world_size):
                self.queue_instruct[device_id].put(("stop"))
                self.procs[device_id].terminate()
