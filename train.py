""" Main training workflow """
#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import argparse
import glob
import os
import sys
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch import cuda

import onmt.inputters as inputters
import onmt.opts as opts
import onmt.modules
from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, _load_fields, _collect_report_features
from onmt.model_builder import build_model
from onmt.utils.optimizers import build_optim
from onmt.utils.misc import use_gpu
from onmt.utils.loss import make_loss_compute

PARSER = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

opts.add_md_help_argument(PARSER)
opts.model_opts(PARSER)
opts.train_opts(PARSER)

OPT = PARSER.parse_args()
if OPT.word_vec_size != -1:
    OPT.src_word_vec_size = OPT.word_vec_size
    OPT.tgt_word_vec_size = OPT.word_vec_size

if OPT.layers != -1:
    OPT.enc_layers = OPT.layers
    OPT.dec_layers = OPT.layers

OPT.brnn = (OPT.encoder_type == "brnn")
if OPT.seed > 0:
    random.seed(OPT.seed)
    torch.manual_seed(OPT.seed)

if OPT.rnn_type == "SRU" and not OPT.gpuid:
    raise AssertionError("Using SRU requires -gpuid set.")

if torch.cuda.is_available() and not OPT.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if OPT.gpuid:
    cuda.set_device(OPT.gpuid[0])
    if OPT.seed > 0:
        torch.cuda.manual_seed(OPT.seed)

if len(OPT.gpuid) > 1:
    sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
    sys.exit(1)

# Set up the Crayon logging server.
if OPT.exp_host != "":
    from pycrayon import CrayonClient

    CC = CrayonClient(hostname=OPT.exp_host)

    EXPERIMENTS = CC.get_experiment_names()
    print(EXPERIMENTS)
    if OPT.exp in EXPERIMENTS:
        CC.remove_experiment(OPT.exp)
    EXPERIMENT = CC.create_experiment(OPT.exp)

if OPT.tensorboard:
    from tensorboardX import SummaryWriter
    WRITER = SummaryWriter(
        OPT.tensorboard_log_dir + datetime.now().strftime("/%b-%d_%H-%M-%S"),
        comment="Unmt")


def report_func(epoch, batch, num_batches,
                progress_step,
                start_time, learning_rate, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        progress_step(int): the progress step.
        start_time(float): last report time.
        learning_rate(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % OPT.report_every == -1 % OPT.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        if OPT.exp_host:
            report_stats.log("progress", EXPERIMENT, learning_rate)
        if OPT.tensorboard:
            # Log the progress using the number of batches on the x-axis.
            report_stats.log_tensorboard(
                "progress", WRITER, learning_rate, progress_step)
        report_stats = onmt.Statistics()

    return report_stats


def train_model(model, fields, optim, data_type, model_opt, opt):
    """
    Train the model.
    """
    train_loss = make_loss_compute(model, fields["tgt"].vocab, opt)
    valid_loss = make_loss_compute(model, fields["tgt"].vocab, opt,
                                   train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count

    trainer = onmt.Trainer(model, train_loss, valid_loss, optim,
                           trunc_size, shard_size, data_type,
                           norm_method, grad_accum_count)

    print('\nStart training...')
    print(' * number of epochs: %d, starting from Epoch %d' %
          (opt.epochs + 1 - opt.start_epoch, opt.start_epoch))
    print(' * batch size: %d' % opt.batch_size)

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        # 1. Train for one epoch on the training set.
        train_iter = build_dataset_iter(lazily_load_dataset("train", opt),
                                       fields, opt)
        train_stats = trainer.train(train_iter, epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validate on the validation set.
        valid_iter = build_dataset_iter(lazily_load_dataset("valid", opt),
                                       fields, opt,
                                       is_train=False)
        valid_stats = trainer.validate(valid_iter)
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # 3. Log to remote server.
        if opt.exp_host:
            train_stats.log("train", EXPERIMENT, optim.lr)
            valid_stats.log("valid", EXPERIMENT, optim.lr)
        if opt.tensorboard:
            train_stats.log_tensorboard("train", WRITER, optim.lr, epoch)
            train_stats.log_tensorboard("valid", WRITER, optim.lr, epoch)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats)



def _check_save_model_path():
    save_model_path = os.path.abspath(OPT.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)



def main():
    # Load checkpoint if we resume from a previous training.
    if OPT.train_from:
        print('Loading checkpoint from %s' % OPT.train_from)
        checkpoint = torch.load(OPT.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear.
        OPT.start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = OPT

    # Peek the fisrt dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train", OPT))
    data_type = first_dataset.data_type

    # Load fields generated from preprocess phase.
    fields = _load_fields(first_dataset, data_type, OPT, checkpoint)

    # Report src/tgt features.
    _collect_report_features(fields)

    # Build model.
    model = build_model(model_opt, OPT, fields, checkpoint)
    _tally_parameters(model)
    _check_save_model_path()

    # Build optimizer.
    optim = build_optim(model, OPT, checkpoint)

    # Do training.
    train_model(model, fields, optim, data_type, model_opt, OPT)

    # If using tensorboard for logging, close the writer after training.
    if OPT.tensorboard:
        WRITER.close()


if __name__ == "__main__":
    main()
