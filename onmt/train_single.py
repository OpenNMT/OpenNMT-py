#!/usr/bin/env python
"""Training on a single process."""
import torch
import sys
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser
from onmt.constants import CorpusTask
from onmt.transforms import make_transforms, save_transforms, \
    get_specials, get_transforms_cls
from onmt.inputters import build_vocab, IterOnDevice
from onmt.inputters.inputter import dict_to_vocabs
from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter
from onmt.inputters.text_corpus import save_transformed_sample
from onmt.model_builder import build_model
from onmt.models.model_saver import load_checkpoint
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.modules.embeddings import prepare_pretrained_embeddings


def prepare_transforms_vocabs(opt):
    """Prepare or dump transforms before training."""
    transforms_cls = get_transforms_cls(opt._all_transform)
    specials = get_specials(opt, transforms_cls)

    vocabs = build_vocab(opt, specials)

    # maybe prepare pretrained embeddings, if any
    prepare_pretrained_embeddings(opt, vocabs)

    if opt.dump_transforms or opt.n_sample != 0:
        transforms = make_transforms(opt, transforms_cls, vocabs)
    if opt.dump_transforms:
        save_transforms(transforms, opt.save_data, overwrite=opt.overwrite)
    if opt.n_sample != 0:
        logger.warning(
            "`-n_sample` != 0: Training will not be started. "
            f"Stop after saving {opt.n_sample} samples/corpus.")
        save_transformed_sample(opt, transforms, n_sample=opt.n_sample)
        logger.info(
            "Sample saved, please check it before restart training.")
        sys.exit()
    return vocabs, transforms_cls


def _init_train(opt):
    """Common initilization stuff for all training process."""
    ArgumentParser.validate_prepare_opts(opt)

    if opt.train_from:
        # Load checkpoint if we resume from a previous training.
        checkpoint = load_checkpoint(ckpt_path=opt.train_from)
        vocabs = dict_to_vocabs(checkpoint['vocab'])
        transforms_cls = get_transforms_cls(opt._all_transform)
        if (hasattr(checkpoint["opt"], '_all_transform') and
                len(opt._all_transform.symmetric_difference(
                    checkpoint["opt"]._all_transform)) != 0):
            _msg = "configured transforms is different from checkpoint:"
            new_transf = opt._all_transform.difference(
                checkpoint["opt"]._all_transform)
            old_transf = checkpoint["opt"]._all_transform.difference(
                opt._all_transform)
            if len(new_transf) != 0:
                _msg += f" +{new_transf}"
            if len(old_transf) != 0:
                _msg += f" -{old_transf}."
            logger.warning(_msg)
        if opt.update_vocab:
            logger.info("Updating checkpoint vocabulary with new vocabulary")
            vocabs, transforms_cls = prepare_transforms_vocabs(opt)
    else:
        checkpoint = None
        vocabs, transforms_cls = prepare_transforms_vocabs(opt)

    return checkpoint, vocabs, transforms_cls


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def _get_model_opts(opt, checkpoint=None):
    """Get `model_opt` to build model, may load from `checkpoint` if any."""
    if checkpoint is not None:
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        if (opt.tensorboard_log_dir == model_opt.tensorboard_log_dir and
                hasattr(model_opt, 'tensorboard_log_dir_dated')):
            # ensure tensorboard output is written in the directory
            # of previous checkpoints
            opt.tensorboard_log_dir_dated = model_opt.tensorboard_log_dir_dated
        # Override checkpoint's update_embeddings as it defaults to false
        model_opt.update_vocab = opt.update_vocab
        # Override checkpoint's freezing settings as it defaults to false
        model_opt.freeze_encoder = opt.freeze_encoder
        model_opt.freeze_decoder = opt.freeze_decoder
    else:
        model_opt = opt
    return model_opt


def _build_valid_iter(opt, transforms_cls, vocabs):
    """Build iterator used for validation."""
    valid_iter = build_dynamic_dataset_iter(
        opt, transforms_cls, vocabs, task=CorpusTask.VALID,
        copy=opt.copy_attn)
    return valid_iter


def _build_train_iter(opt, transforms_cls, vocabs, stride=1, offset=0):
    """Build training iterator."""
    train_iter = build_dynamic_dataset_iter(
        opt, transforms_cls, vocabs, task=CorpusTask.TRAIN,
        copy=opt.copy_attn, stride=stride, offset=offset)
    return train_iter


def main(opt, device_id):
    """Start training on `device_id`."""
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.

    configure_process(opt, device_id)
    init_logger(opt.log_file)

    checkpoint, vocabs, transforms_cls = _init_train(opt)

    model_opt = _get_model_opts(opt, checkpoint=checkpoint)

    # Build model.
    model = build_model(model_opt, opt, vocabs, checkpoint)
    model.count_parameters(log=logger.info)
    logger.info(' * src vocab size = %d' % len(vocabs['src']))
    logger.info(' * tgt vocab size = %d' % len(vocabs['tgt']))
    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, vocabs, optim)

    trainer = build_trainer(
        opt, device_id, model, vocabs, optim, model_saver=model_saver)

    _train_iter = _build_train_iter(opt, transforms_cls, vocabs,
                                    stride=max(1, len(opt.gpu_ranks)),
                                    offset=max(0, device_id))
    train_iter = IterOnDevice(_train_iter, device_id)

    valid_iter = _build_valid_iter(opt, transforms_cls, vocabs)
    if valid_iter is not None:
        valid_iter = IterOnDevice(valid_iter, device_id)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()
