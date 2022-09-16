import codecs
import os
from onmt.utils.parse import ArgumentParser
from onmt.translate import GNMTGlobalScorer, Translator
from onmt.opts import translate_opts
from onmt.constants import DefaultTokens


class Detokenizer():
    """ Allow detokenizing sequences in batchs"""
    def __init__(self, opt):
        if 'bpe' in opt.transforms:
            self.type = "subword-nmt"
        elif 'sentencepiece' in opt.transforms:
            self.type = "sentencepiece"
        elif 'onmt_tokenize' in opt.transforms:
            self.type = "pyonmttok"
            self.tgt_onmttok_kwargs = opt.tgt_onmttok_kwargs
        else:
            self.type = None
        if self.type in ['bpe', 'sentencepiece']:
            if opt.tgt_subword_model is None:
                raise ValueError(
                    "Missing mandatory tokenizer option 'tgt_subword_model'")
            else:
                self.model_path = opt.tgt_subword_model

    def build_detokenizer(self):
        if self.type == "pyonmttok":
            import pyonmttok
            self.tgt_detokenizer = pyonmttok.Tokenizer(
                **self.tgt_onmttok_kwargs)
        elif self.type == "sentencepiece":
            import sentencepiece as spm
            self.tgt_detokenizer = spm.SentencePieceProcessor()
            self.tgt_detokenizer.Load(self.model_path)
        elif self.type == "subword-nmt":
            from subword_nmt.apply_bpe import BPE
            with open(self.model_path, encoding='utf-8') as tgt_codes:
                self.tgt_detokenizer = BPE(codes=tgt_codes, vocab=None)
        else:
            self.tgt_detokenizer = None
        return self.tgt_detokenizer

    def _detokenize(self, tokens):
        if self.type == "pyonmttok":
            detok = self.tgt_detokenizer.detokenize(tokens)
        elif self.type == "sentencepiece":
            detok = self.tgt_detokenizer.DecodePieces(tokens)
        elif self.type == "subword-nmt":
            detok = self.tgt_detokenizer.segment_tokens(tokens, dropout=0.0)
        else:
            detok = " ".join(tokens)
        return detok


class ScoringPreparator():
    """Allow the calculation of metrics via the Trainer's
     training_eval_handler method"""
    def __init__(self, vocabs, opt):
        self.vocabs = vocabs
        self.opt = opt
        self.tgt_detokenizer = Detokenizer(opt)
        self.tgt_detokenizer.build_detokenizer()

    def tokenize_batch(self, batch_side, side):
        """Convert a batch into a list of tokenized sentences"""
        vocab = self.vocabs[side]
        tokenized_sentences = []
        for i in range(batch_side.shape[1]):
            tokens = []
            for t in range(batch_side.shape[0]):
                token = vocab.get_itos()[batch_side[t, i, 0]]
                if (token == DefaultTokens.PAD
                        or token == DefaultTokens.EOS):
                    break
                if token != DefaultTokens.BOS:
                    tokens.append(token)
            tokenized_sentences.append(tokens)
        return tokenized_sentences

    def build_sources_and_refs(self, batch, mode):
        """Reconstruct the sources and references of the examples
        related to a batch"""
        if mode == 'valid':
            sources = []
            refs = []
            for example in batch.dataset.examples:
                sources.append(example.src[0])
                refs.append(example.tgt[0])
        elif mode == 'train':
            sources = self.tokenize_batch(batch.src[0], 'src')
            refs = self.tokenize_batch(batch.tgt, 'tgt')
        return sources, refs

    def translate(self, model, batch, gpu_rank, step, mode):
        """Compute the sentences predicted by the current model's state
        related to a batch"""
        model_opt = self.opt
        parser = ArgumentParser()
        translate_opts(parser)
        base_args = (["-model", "dummy"] + ["-src", "dummy"])
        opt = parser.parse_args(base_args)
        opt.gpu = gpu_rank
        ArgumentParser.validate_translate_opts(opt)
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        scorer = GNMTGlobalScorer.from_opt(opt)
        out_file = codecs.open(os.devnull, "w", "utf-8")
        translator = Translator.from_opt(
            model,
            self.vocabs,
            opt,
            model_opt,
            global_scorer=scorer,
            out_file=out_file,
            report_align=opt.report_align,
            report_score=True,
            logger=None)
        sources, refs = self.build_sources_and_refs(batch, mode)
        _, preds = translator.translate(
            sources,
            batch_size=model_opt.valid_batch_size,
            batch_type=model_opt.batch_type)
        texts_ref = []

        for i in range(len(preds)):
            preds[i] = self.tgt_detokenizer._detokenize(preds[i][0].split())
            texts_ref.append(self.tgt_detokenizer._detokenize(refs[i]))

        if len(preds) > 0 and self.opt.scoring_debug:
            path = os.path.join(self.opt.dump_preds,
                                "preds.{}_step_{}.{}".format(
                                    mode, step, "txt"))
            with open(path, "a") as file:
                for i in range(len(preds)):
                    file.write("SOURCE: {}\n".format(sources[i]))
                    file.write("REF: {}\n".format(texts_ref[i]))
                    file.write("PRED: {}\n\n".format(preds[i]))
        return preds, texts_ref
