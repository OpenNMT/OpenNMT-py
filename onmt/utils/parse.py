import configargparse as cfargparse
import os

import torch

import onmt.opts as opts
from onmt.utils.logging import logger


class ArgumentParser(cfargparse.ArgumentParser):
    def __init__(
            self,
            config_file_parser_class=cfargparse.YAMLConfigFileParser,
            formatter_class=cfargparse.ArgumentDefaultsHelpFormatter,
            **kwargs):
        super(ArgumentParser, self).__init__(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class,
            **kwargs)

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

        if model_opt.layers > 0:
            model_opt.enc_layers = model_opt.layers
            model_opt.dec_layers = model_opt.layers

        if model_opt.rnn_size > 0:
            model_opt.enc_rnn_size = model_opt.rnn_size
            model_opt.dec_rnn_size = model_opt.rnn_size

        model_opt.brnn = model_opt.encoder_type == "brnn"

        if model_opt.copy_attn_type is None:
            model_opt.copy_attn_type = model_opt.global_attention

    @classmethod
    def validate_model_opts(cls, model_opt):
        assert model_opt.model_type in ["text", "img", "audio"], \
            "Unsupported model type %s" % model_opt.model_type

        # this check is here because audio allows the encoder and decoder to
        # be different sizes, but other model types do not yet
        same_size = model_opt.enc_rnn_size == model_opt.dec_rnn_size
        assert model_opt.model_type == 'audio' or same_size, \
            "The encoder and decoder rnns must be the same size for now"

        assert model_opt.rnn_type != "SRU" or model_opt.gpu_ranks, \
            "Using SRU requires -gpu_ranks set."
        if model_opt.share_embeddings:
            if model_opt.model_type != "text":
                raise AssertionError(
                    "--share_embeddings requires --model_type text.")
        if model_opt.model_dtype == "fp16":
            logger.warning(
                "FP16 is experimental, the generated checkpoints may "
                "be incompatible with a future version")

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
        if opt.epochs:
            raise AssertionError(
                "-epochs is deprecated please use -train_steps.")
        if opt.truncated_decoder > 0 and max(opt.accum_count) > 1:
            raise AssertionError("BPTT is not compatible with -accum > 1")
        if opt.gpuid:
            raise AssertionError("gpuid is deprecated \
                  see world_size and gpu_ranks")
        if torch.cuda.is_available() and not opt.gpu_ranks:
            logger.info("WARNING: You have a CUDA device, \
                        should run with -gpu_ranks")

    @classmethod
    def validate_translate_opts(cls, opt):
        if opt.beam_size != 1 and opt.random_sampling_topk != 1:
            raise ValueError('Can either do beam search OR random sampling.')

    @classmethod
    def validate_preprocess_args(cls, opt):
        assert opt.max_shard_size == 0, \
            "-max_shard_size is deprecated. Please use \
            -shard_size (number of examples) instead."
        assert opt.shuffle == 0, \
            "-shuffle is not implemented. Please shuffle \
            your data before pre-processing."

        assert os.path.isfile(opt.train_src) \
            and os.path.isfile(opt.train_tgt), \
            "Please check path of your train src and tgt files!"

        assert not opt.valid_src or os.path.isfile(opt.valid_src), \
            "Please check path of your valid src file!"
        assert not opt.valid_tgt or os.path.isfile(opt.valid_tgt), \
            "Please check path of your valid tgt file!"

        assert not opt.src_vocab or os.path.isfile(opt.src_vocab), \
            "Please check path of your src vocab!"
        assert not opt.tgt_vocab or os.path.isfile(opt.tgt_vocab), \
            "Please check path of your tgt vocab!"
