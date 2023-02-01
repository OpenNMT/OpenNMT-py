import codecs
import os
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
        self.transforms = opt.transforms
        transforms_cls = get_transforms_cls(self.transforms)
        transforms = make_transforms(self.opt, transforms_cls, self.vocabs)
        self.transform = TransformPipe.build_from(transforms.values())

    def warm_up(self, transforms):
        self.transforms = transforms
        transforms_cls = get_transforms_cls(self.transforms)
        transforms = make_transforms(self.opt, transforms_cls, self.vocabs)
        self.transform = TransformPipe.build_from(transforms.values())

    def ids_to_tokens_batch_side(self, batch, side):
        """Convert a batch into a list of transformed sentences
        Args:
            batch: batch yielded from `DynamicDatasetIter` object
            side (string): 'src' or 'tgt'.
        Returns
            transformed_sentences (list): List of lists of tokens.
                Each list is a transformed sentence.
        """
        vocab = self.vocabs[side]
        batch_side = batch[side]
        nb_sentences = batch_side.shape[0]
        nb_tokens_per_sentence = batch_side.shape[1]
        indices_to_remove = [vocab.lookup_token(token)
                             for token in [DefaultTokens.PAD,
                                           DefaultTokens.EOS,
                                           DefaultTokens.BOS]]
        transformed_sentences = []
        for i in range(nb_sentences):
            tokens = [vocab.lookup_index(batch_side[i, t, 0])
                      for t in range(nb_tokens_per_sentence)
                      if batch_side[i, t, 0] not in indices_to_remove]
            transformed_sentences.append(tokens)
        return transformed_sentences

    def ids_to_tokens_batch(self, batch):
        """Reconstruct transformed source and reference
        sentences from a batch.
        Args:
            batch: batch yielded from `DynamicDatasetIter` object
        Returns:
            transformed_batch(list): A list of examples
        with the fields "src" and "tgt"
        """
        transformed_batch = [{'src': src_ex, 'tgt': tgt_ex}
                             for src_ex, tgt_ex
                             in zip(
                                self.ids_to_tokens_batch_side(batch, 'src'),
                                self.ids_to_tokens_batch_side(batch, 'tgt'))]
        return transformed_batch

    def translate(self, model, transformed_batches, gpu_rank, step, mode):
        """Compute and save the sentences predicted by the
        current model's state related to a batch.

        Args:
            model (:obj:`onmt.models.NMTModel`): The current model's state.
            transformed_batches(list of lists): A list of transformed batches.
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
        # translate
        preds = []
        raw_sources = []
        raw_refs = []
        for batch in transformed_batches:
            # for validation we build an infer_iter per batch
            # in order to avoid oom issues because there is no
            # batching strategy in `textbatch_to_tensor`
            numeric = []
            for i, ex in enumerate(batch):
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

        # apply_reverse refs
        if self.transforms:
            texts_ref = [self.transform.apply_reverse(raw_ref)
                         for raw_ref in raw_refs]
            # flatten preds
            preds = [item for preds_ in preds for item in preds_]
        else:
            texts_ref = [" ".join(raw_ref) for raw_ref in raw_refs]
            preds = [" ".join(preds_) for preds_ in preds]

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
        return preds, texts_ref
