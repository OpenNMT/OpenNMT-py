import codecs
import os
import torch
from onmt.utils.parse import ArgumentParser
from onmt.translate import GNMTGlobalScorer, Translator
from onmt.opts import translate_opts
from onmt.constants import DefaultTokens
from onmt.inputters.text_utils import _addcopykeys, tensorify, text_sort_key
from onmt.inputters.inputter import IterOnDevice
from onmt.transforms import get_transforms_cls, make_transforms, TransformPipe


class ScoringPreparator():
    """Allow the calculation of metrics via the Trainer's
     training_eval_handler method.
    """
    def __init__(self, vocabs, opt):
        self.vocabs = vocabs
        self.opt = opt
        if self.opt.dump_preds is not None:
            if not os.path.exists(self.opt.dump_preds):
                os.makedirs(self.opt.dump_preds)
        transforms_cls = get_transforms_cls(opt.transforms)
        transforms = make_transforms(self.opt, transforms_cls, self.vocabs)
        self.transform = TransformPipe.build_from(transforms.values())

    def warm_up(self, transforms):
        transforms_cls = get_transforms_cls(transforms)
        transforms = make_transforms(self.opt, transforms_cls, self.vocabs)
        self.transform = TransformPipe.build_from(transforms.values())

    def tokenize_batch_side(self, batch, side):
        """Convert a batch into a list of tokenized sentences
        Args:
            batch: batch yielded from `DynamicDatasetIter` object
            side (string): 'src' or 'tgt'.
        Returns
            tokenized_sentences (list): List of lists of tokens.
                Each list is a tokenized sentence.
        """
        vocab = self.vocabs[side]
        batch_side = batch[side]
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

    def tokenize_batch(self, batch):
        """Reconstruct raw sources and references from a batch
        Args:
            batch: batch yielded from `DynamicDatasetIter` object
        Returns:
            tokenized_batch(list): A list of examples
        with the fields "src" and "tgt"
        """
        tokenized_batch = [{'src': src_ex, 'tgt': tgt_ex}
                           for src_ex, tgt_ex
                           in zip(self.tokenize_batch_side(batch, 'src'),
                                  self.tokenize_batch_side(batch, 'tgt'))]
        return tokenized_batch

    def translate(self, model, tokenized_batchs, gpu_rank, step, mode):
        """Compute and save the sentences predicted by the
        current model's state related to a batch.

        Args:
            model (:obj:`onmt.models.NMTModel`): The current model's state.
            tokenized_batchs(list of lists): A list of tokenized batchs.
            gpu_rank (int): Ordinal rank of the gpu where the
                translation is to be done.
            step: The current training step.
            mode: (string): 'train' or 'valid'.
        Returns:
            preds (list): Detokenized predictions
            texts_ref (list): Detokenized target sentences
        """
        with open("tokenized_batchs", "w") as f:
            f.write(str(type(tokenized_batchs)) + "/n")
            f.write(str(tokenized_batchs) + "/n")
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
        # translate
        preds = []
        raw_sources = []
        raw_refs = []
        for batch in tokenized_batchs:
            # for validation we build an infer_iter per batch
            # in order to avoid oom issues because there is no
            # batching strategy in `textbatch_to_tensor`
            numeric = []
            for i, ex in enumerate(batch):
                # ex =  self.transform.apply(ex)
                if ex is not None:
                    raw_sources.append(ex['src'])
                    raw_refs.append(ex['tgt'])
                    if isinstance(ex['src'], bytes):
                        ex['src'] = ex['src'].decode("utf-8")
                    idxs = translator.vocabs['src'](ex['src'])
                    num_ex = {'src': {'src': " ".join(ex['src']),
                              'src_ids': idxs},
                              'srclen': len(ex['src']),
                              'tgt': None,
                              'indices': i,
                              'align': None}
                    num_ex = _addcopykeys(translator.vocabs["src"], num_ex)
                    num_ex["src"]["src"] = ex['src']
                    numeric.append(num_ex)
            numeric.sort(key=text_sort_key, reverse=True)
            infer_iter = [tensorify(self.vocabs, numeric)]
            infer_iter = IterOnDevice(infer_iter, opt.gpu)
            _, preds_ = translator._translate(
                        infer_iter, transform=self.transform)
            preds += preds_
        # apply_reverse on raw references
        texts_ref = [self.transform.apply_reverse(raw_ref, "tgt")
                     for raw_ref in raw_refs]
        # apply_reverse on predictions
        preds = [self.transform.apply_reverse(preds_, "tgt")
                 for preds_ in preds]

        # save results
        if len(preds) > 0 and self.opt.scoring_debug:
            path = os.path.join(self.opt.dump_preds,
                                "preds.{}_step_{}.{}".format(
                                    mode, step, "txt"))
            with open(path, "a") as file:
                for i in range(len(preds)):
                    file.write("SOURCE: {}\n".format(raw_sources[i]))
                    file.write("REF: {}\n".format(texts_ref[i]))
                    file.write("PRED: {}\n\n".format(preds[i]))
        # We deactivate the decoder's cache
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
