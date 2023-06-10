import configargparse as cfargparse
import os
import torch

import onmt.opts as opts
from onmt.utils.logging import logger
from onmt.constants import CorpusName, ModelTask
from onmt.transforms import AVAILABLE_TRANSFORMS


class DataOptsCheckerMixin(object):
    """Checker with methods for validate data related options."""

    @staticmethod
    def _validate_file(file_path, info):
        """Check `file_path` is valid or raise `IOError`."""
        if not os.path.isfile(file_path):
            raise IOError(f"Please check path of your {info} file!")

    @classmethod
    def _validate_data(cls, opt):
        """Parse corpora specified in data field of YAML file."""
        import yaml

        default_transforms = opt.transforms
        if len(default_transforms) != 0:
            logger.info(f"Default transforms: {default_transforms}.")
        corpora = yaml.safe_load(opt.data)

        for cname, corpus in corpora.items():
            # Check Transforms
            _transforms = corpus.get("transforms", None)
            if _transforms is None:
                logger.info(
                    f"Missing transforms field for {cname} data, "
                    f"set to default: {default_transforms}."
                )
                corpus["transforms"] = default_transforms
            # Check path
            path_src = corpus.get("path_src", None)
            path_tgt = corpus.get("path_tgt", None)
            if path_src is None:
                raise ValueError(
                    f"Corpus {cname} src path is required."
                    "tgt path is also required for non language"
                    " modeling tasks."
                )
            else:
                opt.data_task = ModelTask.SEQ2SEQ
                if path_tgt is None:
                    logger.debug(
                        "path_tgt is None, it should be set unless the task"
                        " is language modeling"
                    )
                    opt.data_task = ModelTask.LANGUAGE_MODEL
                    # tgt is src for LM task
                    corpus["path_tgt"] = path_src
                    corpora[cname] = corpus
                    path_tgt = path_src
                cls._validate_file(path_src, info=f"{cname}/path_src")
                cls._validate_file(path_tgt, info=f"{cname}/path_tgt")
            path_align = corpus.get("path_align", None)
            if path_align is None:
                if hasattr(opt, "lambda_align") and opt.lambda_align > 0.0:
                    raise ValueError(
                        f"Corpus {cname} alignment file path are "
                        "required when lambda_align > 0.0"
                    )
                corpus["path_align"] = None
            else:
                cls._validate_file(path_align, info=f"{cname}/path_align")
            # Check weight
            weight = corpus.get("weight", None)
            if weight is None:
                if cname != CorpusName.VALID:
                    logger.warning(
                        f"Corpus {cname}'s weight should be given."
                        " We default it to 1 for you."
                    )
                corpus["weight"] = 1

            # Check features
            if opt.n_src_feats > 0:
                if "inferfeats" not in corpus["transforms"]:
                    raise ValueError(
                        "'inferfeats' transform is required "
                        "when setting source features"
                    )

        logger.info(f"Parsed {len(corpora)} corpora from -data.")
        opt.data = corpora

    @classmethod
    def _validate_transforms_opts(cls, opt):
        """Check options used by transforms."""
        for name, transform_cls in AVAILABLE_TRANSFORMS.items():
            if name in opt._all_transform:
                transform_cls._validate_options(opt)

    @classmethod
    def _get_all_transform(cls, opt):
        """Should only called after `_validate_data`."""
        all_transforms = set(opt.transforms)
        for cname, corpus in opt.data.items():
            _transforms = set(corpus["transforms"])
            if len(_transforms) != 0:
                all_transforms.update(_transforms)
        if hasattr(opt, "lambda_align") and opt.lambda_align > 0.0:
            if not all_transforms.isdisjoint({"sentencepiece", "bpe", "onmt_tokenize"}):
                raise ValueError(
                    "lambda_align is not compatible with" " on-the-fly tokenization."
                )
            if not all_transforms.isdisjoint({"tokendrop", "prefix", "bart"}):
                raise ValueError(
                    "lambda_align is not compatible yet with"
                    " potentiel token deletion/addition."
                )
        opt._all_transform = all_transforms

    @classmethod
    def _get_all_transform_translate(cls, opt):
        opt._all_transform = opt.transforms

    @classmethod
    def _validate_vocab_opts(cls, opt, build_vocab_only=False):
        """Check options relate to vocab."""

        if build_vocab_only:
            if not opt.share_vocab:
                assert opt.tgt_vocab, "-tgt_vocab is required if not -share_vocab."
            return
        # validation when train:
        cls._validate_file(opt.src_vocab, info="src vocab")
        if not opt.share_vocab:
            cls._validate_file(opt.tgt_vocab, info="tgt vocab")

        if opt.dump_transforms:
            assert (
                opt.save_data
            ), "-save_data should be set if set \
                -dump_transforms."
        # Check embeddings stuff
        if opt.both_embeddings is not None:
            assert (
                opt.src_embeddings is None and opt.tgt_embeddings is None
            ), "You don't need -src_embeddings or -tgt_embeddings \
                if -both_embeddings is set."

        if any(
            [
                opt.both_embeddings is not None,
                opt.src_embeddings is not None,
                opt.tgt_embeddings is not None,
            ]
        ):
            assert (
                opt.embeddings_type is not None
            ), "You need to specify an -embedding_type!"
            assert (
                opt.save_data
            ), "-save_data should be set if use \
                pretrained embeddings."

    @classmethod
    def _validate_language_model_compatibilities_opts(cls, opt):
        if opt.model_task != ModelTask.LANGUAGE_MODEL:
            return

        logger.info("encoder is not used for LM task")

        assert opt.share_vocab and (
            opt.tgt_vocab is None
        ), "vocab must be shared for LM task"

        assert (
            opt.decoder_type == "transformer"
        ), "Only transformer decoder is supported for LM task"

    @classmethod
    def _validate_source_features_opts(cls, opt):
        if opt.src_feats_defaults is not None:
            assert opt.n_src_feats == len(
                opt.src_feats_defaults.split("￨")
            ), "The number source features defaults does not match \
                -n_src_feats"

    @classmethod
    def validate_prepare_opts(cls, opt, build_vocab_only=False):
        """Validate all options relate to prepare (data/transform/vocab)."""
        if opt.n_sample != 0:
            assert (
                opt.save_data
            ), "-save_data should be set if \
                want save samples."
        cls._validate_data(opt)
        cls._get_all_transform(opt)
        cls._validate_transforms_opts(opt)
        cls._validate_vocab_opts(opt, build_vocab_only=build_vocab_only)
        cls._validate_source_features_opts(opt)

    @classmethod
    def validate_model_opts(cls, opt):
        cls._validate_language_model_compatibilities_opts(opt)


class ArgumentParser(cfargparse.ArgumentParser, DataOptsCheckerMixin):
    """OpenNMT option parser powered with option check methods."""

    def __init__(
        self,
        config_file_parser_class=cfargparse.YAMLConfigFileParser,
        formatter_class=cfargparse.ArgumentDefaultsHelpFormatter,
        **kwargs,
    ):
        super(ArgumentParser, self).__init__(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class,
            **kwargs,
        )

    @classmethod
    def defaults(cls, *args):
        """Get default arguments added to a parser by all ``*args``."""
        dummy_parser = cls()
        for callback in args:
            callback(dummy_parser)
        defaults = dummy_parser.parse_known_args([])[0]
        return defaults

    @classmethod
    def update_model_opts(cls, model_opt):
        if model_opt.word_vec_size > 0:
            model_opt.src_word_vec_size = model_opt.word_vec_size
            model_opt.tgt_word_vec_size = model_opt.word_vec_size

        # Backward compatibility with "fix_word_vecs_*" opts
        if hasattr(model_opt, "fix_word_vecs_enc"):
            model_opt.freeze_word_vecs_enc = model_opt.fix_word_vecs_enc
        if hasattr(model_opt, "fix_word_vecs_dec"):
            model_opt.freeze_word_vecs_dec = model_opt.fix_word_vecs_dec

        if model_opt.layers > 0:
            model_opt.enc_layers = model_opt.layers
            model_opt.dec_layers = model_opt.layers

        if model_opt.hidden_size > 0:
            model_opt.enc_hid_size = model_opt.hidden_size
            model_opt.dec_hid_size = model_opt.hidden_size

        model_opt.brnn = model_opt.encoder_type == "brnn"

        if model_opt.copy_attn_type is None:
            model_opt.copy_attn_type = model_opt.global_attention

        if model_opt.alignment_layer is None:
            model_opt.alignment_layer = -2
            model_opt.lambda_align = 0.0
            model_opt.full_context_alignment = False

    @classmethod
    def validate_model_opts(cls, model_opt):
        assert model_opt.model_type in ["text"], (
            "Unsupported model type %s" % model_opt.model_type
        )

        # encoder and decoder should be same sizes
        same_size = model_opt.enc_hid_size == model_opt.dec_hid_size
        assert same_size, "The encoder and decoder rnns must be the same size for now"

        assert (
            model_opt.rnn_type != "SRU" or model_opt.gpu_ranks
        ), "Using SRU requires -gpu_ranks set."
        if model_opt.share_embeddings:
            if model_opt.model_type != "text":
                raise AssertionError("--share_embeddings requires --model_type text.")
        if model_opt.lambda_align > 0.0:
            assert (
                model_opt.decoder_type == "transformer"
            ), "Only transformer is supported to joint learn alignment."
            assert (
                model_opt.alignment_layer < model_opt.dec_layers
                and model_opt.alignment_layer >= -model_opt.dec_layers
            ), "N° alignment_layer should be smaller than number of layers."
            logger.info(
                "Joint learn alignment at layer [{}] "
                "with {} heads in full_context '{}'.".format(
                    model_opt.alignment_layer,
                    model_opt.alignment_heads,
                    model_opt.full_context_alignment,
                )
            )

        if model_opt.feat_merge == "concat" and model_opt.feat_vec_size > 0:
            assert (
                model_opt.feat_vec_size * model_opt.n_src_feats
            ) + model_opt.src_word_vec_size == model_opt.hidden_size, (
                "(feat_vec_size * n_src_feats) + "
                "src_word_vec_size should be equal to hidden_size with "
                "-feat_merge concat mode."
            )

        if model_opt.position_encoding and model_opt.max_relative_positions != 0:
            raise ValueError(
                "Cannot use absolute and relative position encoding at the"
                "same time. Use either --position_encoding=true for legacy"
                "absolute position encoding or --max_realtive_positions with"
                " -1 for Rotary, or > 0 for Relative Position Representations"
                "as in https://arxiv.org/pdf/1803.02155.pdf"
            )
        if model_opt.multiquery and model_opt.num_kv == 0:
            model_opt.num_kv = 1

    @classmethod
    def ckpt_model_opts(cls, ckpt_opt):
        # Load default opt values, then overwrite with the opts in
        # the checkpoint. That way, if there are new options added,
        # the defaults are used.
        opt = cls.defaults(opts.model_opts)
        opt.__dict__.update(ckpt_opt.__dict__)
        return opt

    @classmethod
    def validate_train_opts(cls, opt):
        if torch.cuda.is_available() and not opt.gpu_ranks:
            logger.warn("You have a CUDA device, should run with -gpu_ranks")
        if opt.world_size < len(opt.gpu_ranks):
            raise AssertionError(
                "parameter counts of -gpu_ranks must be less or equal "
                "than -world_size."
            )
        if opt.world_size == len(opt.gpu_ranks) and min(opt.gpu_ranks) > 0:
            raise AssertionError(
                "-gpu_ranks should have master(=0) rank "
                "unless -world_size is greater than len(gpu_ranks)."
            )

        assert len(opt.dropout) == len(
            opt.dropout_steps
        ), "Number of dropout values must match accum_steps values"

        assert len(opt.attention_dropout) == len(
            opt.dropout_steps
        ), "Number of attention_dropout values must match accum_steps values"

        assert len(opt.accum_count) == len(
            opt.accum_steps
        ), "Number of accum_count values must match number of accum_steps"

        if opt.update_vocab:
            assert opt.train_from, "-update_vocab needs -train_from option"
            assert opt.reset_optim in [
                "states",
                "all",
            ], '-update_vocab needs -reset_optim "states" or "all"'

    @classmethod
    def validate_translate_opts(cls, opt):
        if opt.gold_align:
            assert opt.report_align, "-report_align should be enabled with -gold_align"
            assert (
                not opt.replace_unk
            ), "-replace_unk option can not be used with -gold_align enabled"
            assert opt.tgt, "-tgt should be specified with -gold_align"

    @classmethod
    def validate_translate_opts_dynamic(cls, opt):
        # It comes from training
        # TODO: needs to be added as inference opt
        opt.share_vocab = False
