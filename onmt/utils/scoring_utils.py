import codecs
import os
from onmt.utils.parse import ArgumentParser
from onmt.translate import GNMTGlobalScorer, Translator
from onmt.opts import translate_opts
from onmt.constants import CorpusTask
from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter
from onmt.transforms import get_transforms_cls, make_transforms, TransformPipe


class ScoringPreparator:
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

    def translate(self, model, gpu_rank, step):
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
        # ########## #
        # Translator #
        # ########## #

        # Set "default" translation options on empty cfgfile
        parser = ArgumentParser()
        translate_opts(parser)
        base_args = ["-model", "dummy"] + ["-src", "dummy"]
        opt = parser.parse_args(base_args)
        opt.gpu = gpu_rank
        if hasattr(self.opt, "tgt_file_prefix"):
            opt.tgt_file_prefix = self.opt.tgt_file_prefix
        opt.beam_size = 1  # prevent OOM when GPU is almost full at training
        ArgumentParser.validate_translate_opts(opt)

        # Build translator from options
        scorer = GNMTGlobalScorer.from_opt(opt)
        out_file = codecs.open(os.devnull, "w", "utf-8")
        model_opt = self.opt
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        translator = Translator.from_opt(
            model,
            self.vocabs,
            opt,
            model_opt,
            global_scorer=scorer,
            out_file=out_file,
            report_align=opt.report_align,
            report_score=False,
            logger=None,
        )

        # ################### #
        # Validation iterator #
        # ################### #

        # Reinstantiate the validation iterator

        transforms_cls = get_transforms_cls(model_opt._all_transform)
        model_opt.num_workers = 0
        model_opt.tgt = None

        # Retrieve raw references and sources
        with codecs.open(
            model_opt.data["valid"]["path_tgt"], "r", encoding="utf-8"
        ) as f:
            raw_refs = [line.strip("\n") for line in f if line.strip("\n")]
        with codecs.open(
            model_opt.data["valid"]["path_src"], "r", encoding="utf-8"
        ) as f:
            raw_srcs = [line.strip("\n") for line in f if line.strip("\n")]

        valid_iter = build_dynamic_dataset_iter(
            model_opt,
            transforms_cls,
            translator.vocabs,
            task=CorpusTask.VALID,
            tgt="",  # This force to clear the target side (needed when using tgt_file_prefix)
            copy=model_opt.copy_attn,
            device_id=opt.gpu,
        )

        # ########### #
        # Predictions #
        # ########### #

        _, preds = translator._translate(
            valid_iter,
            transform=valid_iter.transforms,
            attn_debug=opt.attn_debug,
            align_debug=opt.align_debug,
        )

        # ####### #
        # Outputs #
        # ####### #

        # Flatten predictions
        preds = [x.lstrip() for sublist in preds for x in sublist]

        # Save results
        if len(preds) > 0 and self.opt.scoring_debug:
            path = os.path.join(self.opt.dump_preds, f"preds.valid_step_{step}.txt")
            with open(path, "a") as file:
                for i in range(len(preds)):
                    file.write("SOURCE: {}\n".format(raw_srcs[i]))
                    file.write("REF: {}\n".format(raw_refs[i]))
                    file.write("PRED: {}\n\n".format(preds[i]))
        return preds, raw_refs
