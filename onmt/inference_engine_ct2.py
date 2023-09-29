import ctranslate2
import pyonmttok
from onmt.constants import CorpusTask, DefaultTokens
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
        self.device_id = 0

    def warm_up(self):
        self.build_generator()
        self.build_tokenizer()

    def build_generator(self):
        self.generator = ctranslate2.Generator(self.opt.models[0], device="cuda")

    def build_tokenizer(self):
        if self.transforms_cls.get('sentencepiece', None) is not None:
            from onmt.transforms.tokenize import SentencePieceTransform
            self.tokenizer = SentencePieceTransform(self.opt)
            self.tokenizer.warm_up()
            self.build_sentencepiece_vocab()
        else:
            logger.error("Only the sentencepiece tokenize transform is supported.")

    def build_sentencepiece_vocab(self):
        n_words = self.tokenizer.load_models["src"].vocab_size()
        vocab = [self.tokenizer.load_models["src"].id_to_piece(i) for i in range(n_words)]
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

    def continue_batch(self, batch, add_bos=True):
        for i in range(batch["src"].size()[0]):
            prompt_ids = batch["src"][i, :, 0].cpu().numpy().tolist()
            prompt_tokens = [self.vocabs['src'].lookup_index(id)
                             for id in prompt_ids]
            if add_bos:
                prompt_tokens.insert(0, "<s>")

            step_results = self.generator.generate_tokens(
                prompt_tokens, sampling_temperature=0.1, sampling_topk=40, max_length=512
            )
            output_ids = []
            for step_result in step_results:
                is_new_word = step_result.token.startswith("▁")

                if is_new_word and output_ids:
                    yield " " + self.tokenizer._detokenize(output_ids)
                    output_ids = []

                output_ids.append(step_result.token_id)

            if output_ids:
                yield " " + self.tokenizer._detokenize(output_ids)

    def infer_list(self, src):
        infer_iter = build_dynamic_dataset_iter(
            self.opt,
            self.transforms_cls,
            self.vocabs,
            task=CorpusTask.INFER,
            src=src,
        )
        infer_iter = IterOnDevice(infer_iter, self.device_id)
        out = []
        for batch in infer_iter:
            words = []
            for _out in self.continue_batch(batch):
                words.append(_out)
            out.append("".join(words))
        return out
