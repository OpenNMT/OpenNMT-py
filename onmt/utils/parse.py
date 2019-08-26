import configargparse as cfargparse
import os

import torch

import onmt.opts as opts
from onmt.utils.logging import logger
from onmt.utils.bert_tokenization import PRETRAINED_VOCAB_ARCHIVE_MAP


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
        assert model_opt.model_type in ["text", "img", "audio", "vec"], \
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
        if opt.is_bert:
            logger.info("WE ARE IN BERT MODE.")
            if opt.task_type is "none":
                raise ValueError(
                    "Downstream task should be chosen when use BERT.")
            if opt.reuse_embeddings is True:
                if opt.task_type != "pretraining":
                    opt.reuse_embeddings = False
                    logger.warning(
                        "reuse_embeddings not available for this task.")
        if opt.epochs:
            raise AssertionError(
                  "-epochs is deprecated please use -train_steps.")
        if opt.truncated_decoder > 0 and max(opt.accum_count) > 1:
            raise AssertionError("BPTT is not compatible with -accum > 1")

        if opt.gpuid:
            raise AssertionError(
                  "gpuid is deprecated see world_size and gpu_ranks")
        if torch.cuda.is_available() and not opt.gpu_ranks:
            logger.info("WARNING: You have a CUDA device, \
                        should run with -gpu_ranks")
        if opt.world_size < len(opt.gpu_ranks):
            raise AssertionError(
                  "parameter counts of -gpu_ranks must be less or equal "
                  "than -world_size.")
        if opt.world_size == len(opt.gpu_ranks) and \
                min(opt.gpu_ranks) > 0:
            raise AssertionError(
                  "-gpu_ranks should have master(=0) rank "
                  "unless -world_size is greater than len(gpu_ranks).")
        assert len(opt.data_ids) == len(opt.data_weights), \
            "Please check -data_ids and -data_weights options!"

        assert len(opt.dropout) == len(opt.dropout_steps), \
            "Number of dropout values must match accum_steps values"

        assert len(opt.attention_dropout) == len(opt.dropout_steps), \
            "Number of attention_dropout values must match accum_steps values"

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

        assert len(opt.train_src) == len(opt.train_tgt), \
            "Please provide same number of src and tgt train files!"

        assert len(opt.train_src) == len(opt.train_ids), \
            "Please provide proper -train_ids for your data!"

        for file in opt.train_src + opt.train_tgt:
            assert os.path.isfile(file), "Please check path of %s" % file

        assert not opt.valid_src or os.path.isfile(opt.valid_src), \
            "Please check path of your valid src file!"
        assert not opt.valid_tgt or os.path.isfile(opt.valid_tgt), \
            "Please check path of your valid tgt file!"

        assert not opt.src_vocab or os.path.isfile(opt.src_vocab), \
            "Please check path of your src vocab!"
        assert not opt.tgt_vocab or os.path.isfile(opt.tgt_vocab), \
            "Please check path of your tgt vocab!"

    @classmethod
    def validate_preprocess_bert_opts(cls, opt):
        assert opt.vocab_model in PRETRAINED_VOCAB_ARCHIVE_MAP.keys(), \
            "Unsupported Pretrain model '%s'" % (opt.vocab_model)
        if '-cased' in opt.vocab_model and opt.do_lower_case is True:
            logger.warning("The pre-trained model you are loading is " +
                           "cased model, you shouldn't set `do_lower_case`," +
                           "we turned it off for you.")
            opt.do_lower_case = False
        elif '-cased' not in opt.vocab_model and opt.do_lower_case is False:
            logger.warning("The pre-trained model you are loading is " +
                           "uncased model, you should set `do_lower_case`, " +
                           "we turned it on for you.")
            opt.do_lower_case = True

        for filename in opt.data:
            assert os.path.isfile(filename),\
                "Please check path of %s" % filename

        if opt.task == "tagging":
            assert opt.file_type == 'txt' and len(opt.data) == 1,\
                "For sequence tagging, only single txt file is supported."
            opt.data = opt.data[0]

            assert len(opt.input_columns) == 1,\
                "For sequence tagging, only one column for input tokens."
            opt.input_columns = opt.input_columns[0]

            assert opt.label_column is not None,\
                "For sequence tagging, label column should be given."

        if opt.task == "classification":
            if opt.file_type == "csv":
                assert len(opt.data) == 1,\
                    "For csv, only single file is needed."
                opt.data = opt.data[0]
                assert len(opt.input_columns) in [1, 2],\
                    "Please indicate colomn of sentence A (and B)"
                assert opt.label_column is not None,\
                    "For csv file, label column should be given."
                if opt.delimiter != '\t':
                    logger.warning("for csv file, we set delimiter to '\t'")
                    opt.delimiter = '\t'
        return opt

    @classmethod
    def validate_predict_opts(cls, opt):
        if opt.delimiter is None:
            if opt.task == 'classification':
                opt.delimiter = ' ||| '
            else:
                opt.delimiter = ' '
        logger.info("NOTICE: opt.delimiter set to `%s`" % opt.delimiter)
        assert opt.vocab_model in PRETRAINED_VOCAB_ARCHIVE_MAP.keys(), \
            "Unsupported Pretrain model '%s'" % (opt.vocab_model)
        if '-cased' in opt.vocab_model and opt.do_lower_case is True:
            logger.info("WARNING: The pre-trained model you are loading " +
                        "is cased model, you shouldn't set `do_lower_case`," +
                        "we turned it off for you.")
            opt.do_lower_case = False
        elif '-cased' not in opt.vocab_model and opt.do_lower_case is False:
            logger.info("WARNING: The pre-trained model you are loading " +
                        "is uncased model, you should set `do_lower_case`, " +
                        "we turned it on for you.")
            opt.do_lower_case = True
        return opt
