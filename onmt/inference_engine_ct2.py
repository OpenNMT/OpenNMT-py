import ctranslate2
import pyonmttok
from onmt.constants import CorpusTask, DefaultTokens, ModelTask
from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter
from onmt.inputters.inputter import IterOnDevice
from onmt.transforms import get_transforms_cls
from onmt.utils.logging import logger


class InferenceEngineCT2(object):

    """Wrapper Class to run Inference with ctranslate2.

    Args:
        opt: inference options
    """

    def __init__(self, opt):
        self.opt = opt
        self.transforms_cls = get_transforms_cls(self.opt._all_transform)
        self.logger = logger
        if opt.world_size == 1:
            self.device_id = 0
        # Build translator
        if opt.model_task == ModelTask.LANGUAGE_MODEL:
            self.translator = ctranslate2.Generator(
                opt.models[0], device="cuda", device_index=opt.gpu_ranks
            )
        else:
            self.translator = ctranslate2.Translator(
                self.opt.models[0], device="cuda", device_index=opt.gpu_ranks
            )
        # Build tokenizer and vocab
        if self.transforms_cls.get("sentencepiece", None) is not None:
            from onmt.transforms.tokenize import SentencePieceTransform

            self.tokenizer = SentencePieceTransform(self.opt)
            self.tokenizer.warm_up()
            n_words = self.tokenizer.load_models["src"].vocab_size()
            vocab = [
                self.tokenizer.load_models["src"].id_to_piece(i) for i in range(n_words)
            ]
            vocabs = {}
            vocab[3] = DefaultTokens.PAD
            src_vocab = pyonmttok.build_vocab_from_tokens(
                vocab, maximum_size=n_words, special_tokens=["<unk>", "<s>", "</s>"]
            )
            vocabs["src"] = src_vocab
            vocabs["tgt"] = src_vocab
            vocabs["data_task"] = "lm"
            vocabs["decoder_start_token"] = "<s>"
            self.vocabs = vocabs

    def _translate(self, infer_iter, opt, add_bos=True):
        scores = []
        preds = []
        for batch in infer_iter:
            _scores, _preds = self.translate_batch(batch, opt, add_bos)
            scores += _scores
            preds += _preds
        return scores, preds

    def translate_batch(self, batch, opt, add_bos=True):
        if opt.model_task == ModelTask.LANGUAGE_MODEL:
            start_tokens = []
            for i in range(batch["src"].size()[0]):
                start_ids = batch["src"][i, :, 0].cpu().numpy().tolist()
                _start_tokens = [
                    self.vocabs["src"].lookup_index(id)
                    for id in start_ids
                    if id != self.vocabs["src"].lookup_token(DefaultTokens.PAD)
                ]
                start_tokens.append(_start_tokens)
            translated_batch = self.translator.generate_batch(
                start_tokens=start_tokens,
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
            scores, preds = (
                [out.scores for out in translated_batch],
                [
                    [self.tokenizer._detokenize(tokens) for tokens in out.sequences]
                    for out in translated_batch
                ],
            )
            scores = sum(scores, [])
            preds = sum(preds, [])
        return scores, preds

    def infer_list(self, src):
        if self.opt.world_size == 1:
            infer_iter = build_dynamic_dataset_iter(
                self.opt,
                self.transforms_cls,
                self.vocabs,
                task=CorpusTask.INFER,
                src=src,
            )
            infer_iter = IterOnDevice(infer_iter, self.device_id)
            scores, preds = self._translate(infer_iter, self.opt)
        return scores, preds

    def infer_file(self):
        """File inference. Source file must be the opt.src argument"""
        if self.opt.world_size == 1:
            infer_iter = build_dynamic_dataset_iter(
                self.opt,
                self.transforms_cls,
                self.vocabs,
                task=CorpusTask.INFER,
            )
            infer_iter = IterOnDevice(infer_iter, self.device_id)
            scores, preds = self._translate(infer_iter, self.opt)
            return scores, preds
