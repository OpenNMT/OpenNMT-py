#!/usr/bin/env python
"""Train models with dynamic data."""
import sys
import torch
from functools import partial

# import onmt.opts as opts
from onmt.utils.distributed import ErrorHandler, consumer, batch_producer
from onmt.utils.misc import set_random_seed
from onmt.modules.embeddings import prepare_pretrained_embeddings
from onmt.utils.logging import init_logger, logger

from onmt.models.model_saver import load_checkpoint
from onmt.train_single import main as single_main, _build_train_iter

from onmt.utils.parse import ArgumentParser
from onmt.opts import train_opts
from onmt.inputters.corpus import save_transformed_sample
from onmt.inputters.fields import build_dynamic_fields, save_fields, \
    load_fields
from onmt.transforms import make_transforms, save_transforms, \
    get_specials, get_transforms_cls

# Set sharing strategy manually instead of default based on the OS.
torch.multiprocessing.set_sharing_strategy('file_system')


def prepare_fields_transforms(opt):
    """Prepare or dump fields & transforms before training."""
    transforms_cls = get_transforms_cls(opt._all_transform)
    specials = get_specials(opt, transforms_cls)

    fields = build_dynamic_fields(
        opt, src_specials=specials['src'], tgt_specials=specials['tgt'])

    # maybe prepare pretrained embeddings, if any
    prepare_pretrained_embeddings(opt, fields)

    if opt.dump_fields:
        save_fields(fields, opt.save_data, overwrite=opt.overwrite)
    if opt.dump_transforms or opt.n_sample != 0:
        transforms = make_transforms(opt, transforms_cls, fields)
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
    return fields, transforms_cls


def _init_train(opt):
    """Common initilization stuff for all training process."""
    ArgumentParser.validate_prepare_opts(opt)

    if opt.train_from:
        # Load checkpoint if we resume from a previous training.
        checkpoint = load_checkpoint(ckpt_path=opt.train_from)
        fields = load_fields(opt.save_data, checkpoint)
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
    else:
        checkpoint = None
        fields, transforms_cls = prepare_fields_transforms(opt)

    # Report src and tgt vocab sizes
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))
    return checkpoint, fields, transforms_cls


def train(opt):
    init_logger(opt.log_file)
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    set_random_seed(opt.seed, False)

    checkpoint, fields, transforms_cls = _init_train(opt)
    train_process = partial(
        single_main,
        fields=fields,
        transforms_cls=transforms_cls,
        checkpoint=checkpoint)

    nb_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:

        queues = []
        mp = torch.multiprocessing.get_context('spawn')
        semaphore = mp.Semaphore(opt.world_size * opt.queue_size)
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            q = mp.Queue(opt.queue_size)
            queues += [q]
            procs.append(mp.Process(target=consumer, args=(
                train_process, opt, device_id, error_queue, q, semaphore),
                daemon=True))
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        producers = []
        # This does not work if we merge with the first loop, not sure why
        for device_id in range(nb_gpu):
            # Get the iterator to generate from
            train_iter = _build_train_iter(
                opt, fields, transforms_cls, stride=nb_gpu, offset=device_id)
            producer = mp.Process(target=batch_producer,
                                  args=(train_iter, queues[device_id],
                                        semaphore, opt,),
                                  daemon=True)
            producers.append(producer)
            producers[device_id].start()
            logger.info(" Starting producer process pid: {}  ".format(
                producers[device_id].pid))
            error_handler.add_child(producers[device_id].pid)

        for p in procs:
            p.join()
        # Once training is done, we can terminate the producers
        for p in producers:
            p.terminate()

    elif nb_gpu == 1:  # case 1 GPU only
        train_process(opt, device_id=0)
    else:   # case only CPU
        train_process(opt, device_id=-1)


def _get_parser():
    parser = ArgumentParser(description='train.py')
    train_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt, unknown = parser.parse_known_args()
    train(opt)


if __name__ == "__main__":
    main()
