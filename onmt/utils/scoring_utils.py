import codecs
import os
import torch
from onmt.utils.parse import ArgumentParser
from onmt.translate import GNMTGlobalScorer, Translator
from onmt.opts import translate_opts
from onmt.constants import DefaultTokens
from onmt.inputters.text_utils import _addcopykeys, tensorify, text_sort_key
from onmt.inputters.inputter import IterOnDevice


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
        if self.opt.dump_preds is not None:
            if not os.path.exists(self.opt.dump_preds):
                os.makedirs(self.opt.dump_preds)

    def tokenize_batch(self, batch_side, side):
        """Convert a batch into a list of tokenized sentences
        Args:
            batch_side: 'src' or 'tgt' field of a batch.
            side (string): 'src' or 'tgt'.
        Returns
            tokenized_sentences (list): List of lists of tokens.
                Each list is a tokenized sentence.
        """
        vocab = self.vocabs[side]
        nb_sentences = batch_side.shape[0]
        nb_tokens_per_sentence = batch_side.shape[1]
        indices_to_remove = [vocab.lookup_token(token)
                             for token in [DefaultTokens.PAD,
                                           DefaultTokens.EOS,
                                           DefaultTokens.BOS]]
        tokenized_sentences = []
        for i in range(nb_sentences):
            tokens = [vocab.lookup_index(batch_side[i, t, 0])
                      for t in range(nb_tokens_per_sentence)
                      if batch_side[i, t, 0] not in indices_to_remove]
            tokenized_sentences.append(tokens)
        return tokenized_sentences

    def build_sources_and_refs(self, batch):
        """Reconstruct sources and references from a batch
        Args:
            batch: batch yielded from `DynamicDatasetIter` object
        Returns:
           sources (list): Tokenized source sentences
           refs (list): Tokenized target sentences
        """
        sources = self.tokenize_batch(batch['src'], 'src')
        refs = self.tokenize_batch(batch['tgt'], 'tgt')
        return sources, refs

    def translate(self, model, sources, refs, gpu_rank, step, mode):
        """Compute and save the sentences predicted by the
        current model's state related to a batch.

        Args:
            model (:obj:`onmt.models.NMTModel`): The current model's state.
            sources: (list) List of lists of tokenized source sentences.
                Each list is related to a batch.
            refs (list): Tokenized target sentences.
            gpu_rank (int): Ordinal rank of the gpu where the
                translation is to be done.
            step: The current training step.
            mode: (string): 'train' or 'valid'.
        Returns:
            preds (list): Detokenized predictions
            texts_ref (list): Detokenized target sentences
        """
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
            report_score=False,
            logger=None)
        preds = []
        for sources_ in sources:
            # for validation we build an infer_iter per batch
            # in order to avoid oom issues because there is no
            # batching strategy in `textbatch_to_tensor`
            numeric = []
            for i, ex in enumerate(sources_):
                if isinstance(ex, bytes):
                    ex = ex.decode("utf-8")
                idxs = translator.vocabs['src'](ex)
                num_ex = {'src': {'src': " ".join(ex),
                          'src_ids': idxs},
                          'srclen': len(ex),
                          'tgt': None,
                          'indices': i,
                          'align': None}
                num_ex = _addcopykeys(translator.vocabs["src"], num_ex)
                num_ex["src"]["src"] = ex
                numeric.append(num_ex)
            numeric.sort(key=text_sort_key, reverse=True)
            infer_iter = [tensorify(self.vocabs, numeric)]
            infer_iter = IterOnDevice(infer_iter, opt.gpu)
            _, preds_ = translator._translate(
                        infer_iter)
            preds += preds_
        # flatten sources (for validation)
        sources = [item for sources_ in sources for item in sources_]
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
        # we deactivate the decoder's cache
        # as we use teacher forcing at training time.
        if hasattr(model.decoder, 'transformer_layers'):
            for layer in model.decoder.transformer_layers:
                layer.self_attn.layer_cache = (False,
                                               {'keys': torch.tensor([]),
                                                'values': torch.tensor([])})
                layer.context_attn.layer_cache = (False,
                                                  {'keys': torch.tensor([]),
                                                   'values': torch.tensor([])})
        return preds, texts_ref
