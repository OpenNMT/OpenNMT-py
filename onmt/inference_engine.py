import json
from onmt.constants import CorpusTask, DefaultTokens, ModelTask
from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter
from onmt.inputters.inputter import IterOnDevice
from onmt.utils.distributed import ErrorHandler, spawned_infer
from onmt.utils.logging import logger
from onmt.transforms import get_transforms_cls, make_transforms, TransformPipe


class InferenceEngine(object):
    """Wrapper Class to run Inference.

    Args:
        opt: inference options
    """

    def __init__(self, opt):
        print("###")
        self.opt = opt

    def translate_batch(self, batch):
        pass

    # @classmethod
    def _translate(self, infer_iter):
        pass

    # @classmethod
    def infer_file(self):
        """File inference. Source file must be the opt.src argument"""
        assert self.opt.world_size == 1, 'World size must be equal to 1.'
        infer_iter = build_dynamic_dataset_iter(
            self.opt,
            self.transforms_cls,
            self.vocabs,
            task=CorpusTask.INFER,
        )
        infer_iter = IterOnDevice(infer_iter, self.device_id)
        scores, preds = self._translate(infer_iter)
        return scores, preds

    # @classmethod
    def infer_list(self, src):
        """List of strings inference `src`"""
        assert self.opt.world_size == 1, 'World size must be equal to 1.'
        infer_iter = build_dynamic_dataset_iter(
            self.opt,
            self.transforms_cls,
            self.vocabs,
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

    def infer_file_parallel(self):
        """File inference in mulitprocessing with partitioned models."""
        pass

    def infer_list_parallel(self, src):
        """List of strings inference in mulitprocessing with partitioned models."""
        pass

    def terminate(self):
        pass


class InferenceEnginePY(InferenceEngine):
    """Inference engine subclass to run inference with `translate.py`.

    Args:
        opt: inference options
    """

    def __init__(self, opt):
        import torch
        from onmt.translate.translator import build_translator
        super().__init__(opt)
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
            self.vocabs = self.translator.vocabs

    def _translate(self, infer_iter):
        scores, preds = self.translator._translate(
            infer_iter,
            infer_iter.transform,
            self.opt.attn_debug,
            self.opt.align_debug)
        return scores, preds

    def infer_file_parallel(self):
        assert self.opt.world_size > 1, 'World size must be greater than 1.'
        for device_id in range(self.opt.world_size):
            self.queue_instruct[device_id].put(("infer_file", self.opt))
        scores, preds = [], []
        for device_id in range(self.opt.world_size):
            scores.append(self.queue_result[device_id].get())
            preds.append(self.queue_result[device_id].get())
        return scores[0], preds[0]

    def infer_list_parallel(self, src):
        assert self.opt.world_size > 1, 'World size must be greater than 1.'
        for device_id in range(self.opt.world_size):
            self.queue_instruct[device_id].put(("infer_list", src))
        scores, preds = [], []
        for device_id in range(self.opt.world_size):
            scores.append(self.queue_result[device_id].get())
            preds.append(self.queue_result[device_id].get())
        return scores[0], preds[0]

    def terminate(self):
        assert self.opt.world_size > 1, 'World size must be greater than 1.'
        if self.opt.world_size > 1:
            for device_id in range(self.opt.world_size):
                self.queue_instruct[device_id].put(("stop"))
                self.procs[device_id].terminate()


class InferenceEngineCT2(InferenceEngine):

    """Inference engine subclass to run inference with ctranslate2.

    Args:
        opt: inference options
    """

    def __init__(self, opt):
        import ctranslate2
        import pyonmttok
        super().__init__(opt)
        self.opt = opt
        self.logger = logger
        assert self.opt.world_size == 1, 'World size must be equal to 1.'
        self.device_id = 0
        self.transforms_cls = get_transforms_cls(self.opt._all_transform)
        # Build translator
        if opt.model_task == ModelTask.LANGUAGE_MODEL:
            self.translator = ctranslate2.Generator(
                opt.models[0], device="cuda", device_index=opt.gpu_ranks
            )
        else:
            self.translator = ctranslate2.Translator(
                self.opt.models[0], device="cuda", device_index=opt.gpu_ranks
            )
        # Build vocab
        vocab_path = opt.src_subword_vocab
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        vocabs = {}
        src_vocab = pyonmttok.build_vocab_from_tokens(vocab)
        vocabs["src"] = src_vocab
        vocabs["tgt"] = src_vocab
        vocabs["data_task"] = "lm"
        vocabs["decoder_start_token"] = "<s>"
        self.vocabs = vocabs
        # Build transform pipe
        transforms = make_transforms(opt, self.transforms_cls, self.vocabs)
        self.transform = TransformPipe.build_from(transforms.values())

    def translate_batch(self, batch, opt):
        input_tokens = []
        for i in range(batch["src"].size()[0]):
            start_ids = batch["src"][i, :, 0].cpu().numpy().tolist()
            _input_tokens = [
                self.vocabs["src"].lookup_index(id)
                for id in start_ids
                if id != self.vocabs["src"].lookup_token(DefaultTokens.PAD)
            ]
            input_tokens.append(_input_tokens)
        if opt.model_task == ModelTask.LANGUAGE_MODEL:
            translated_batch = self.translator.generate_batch(
                start_tokens=input_tokens,
                batch_type=("examples" if opt.batch_type == "sents" else "tokens"),
                max_batch_size=opt.batch_size,
                beam_size=opt.beam_size,
                min_length=0,
                max_length=opt.max_length,
                return_scores=True,
                include_prompt_in_result=False,
                sampling_topk=opt.random_sampling_topk,
                sampling_topp=opt.random_sampling_topp,
                sampling_temperature=opt.random_sampling_temp,
            )
            preds = sum(
                [
                    [self.transform.apply_reverse(tokens) for tokens in out.sequences]
                    for out in translated_batch
                ],
                [],
            )
            scores = sum([out.scores for out in translated_batch], [])
        elif opt.model_task == ModelTask.SEQ2SEQ:
            translated_batch = self.translator.translate_batch(
                input_tokens,
                batch_type=("examples" if opt.batch_type == "sents" else "tokens"),
                max_batch_size=opt.batch_size,
                max_decoding_length=opt.max_length,
                return_scores=True,
                sampling_topk=opt.random_sampling_topk,
                sampling_topp=opt.random_sampling_topp,
                sampling_temperature=opt.random_sampling_temp,
            )
            preds = sum(
                [
                    [self.transform.apply_reverse(tokens) for tokens in out.hypotheses]
                    for out in translated_batch
                ],
                [],
            )
            scores = sum([out.scores for out in translated_batch], [])

        return scores, preds

    def _translate(self, infer_iter):
        scores = []
        preds = []
        for batch in infer_iter:
            _scores, _preds = self.translate_batch(batch, self.opt)
            scores += _scores
            preds += _preds
        return scores, preds
