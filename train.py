#!/usr/bin/env python

from __future__ import division

import argparse
import glob
import os
import sys
import random

import torch
import torch.nn as nn
from torch import cuda

import onmt
import onmt.io
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu
import opts


parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)

opt = parser.parse_args()
if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

if opt.rnn_type == "SRU" and not opt.gpuid:
    raise AssertionError("Using SRU requires -gpuid set.")

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

if len(opt.gpuid) > 1:
    sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
    sys.exit(1)


# Set up the Crayon logging server.
if opt.exp_host != "":
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=opt.exp_host)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.exp in experiments:
        cc.remove_experiment(opt.exp)
    experiment = cc.create_experiment(opt.exp)


def report_func(epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch+1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)
        report_stats = onmt.Statistics()

    return report_stats


class LazyIter(object):
    def __init__(self, path, is_train):
        self.path = path
        self.is_train = is_train
        self.fields = None

    def load_iterator(self):
        """
        This returns user-defined train data iterator for the trainer
        to iterate over during each train epoch. We implement simple
        ordered iterator strategy here, but more sophisticated strategy
        like curriculum learning is ok too.
        """
        if self.is_train:
            train_dataset = torch.load(self.path)
            train_dataset.fields = self.fields
            # Sort batch by decreasing lengths of sentence required by pytorch.
            # sort=False means "Use dataset's sortkey instead of iterator's".
            return onmt.io.OrderedIterator(
                dataset=train_dataset, batch_size=opt.batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                sort=False, sort_within_batch=True, repeat=False), \
                train_dataset
        else:
            # Sort batch by decreasing lengths of sentence required by pytorch.
            # sort=False means "Use dataset's sortkey instead of iterator's".
            valid_dataset = torch.load(self.path)
            valid_dataset.fields = self.fields
            return onmt.io.OrderedIterator(
                dataset=valid_dataset, batch_size=opt.valid_batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                train=False, sort=False, sort_within_batch=True), valid_dataset


def make_loss_compute(model, tgt_vocab, opt):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, None, opt.copy_attn_force)
    else:
        compute = onmt.Loss.NMTLossCompute(model.generator, tgt_vocab,
                                           opt.label_smoothing)

    if use_gpu(opt):
        compute.cuda()

    return compute


def train_model(model, train_dataset, valid_dataset, data_type,
                fields, optim, model_opt):

    train_iter = train_dataset
    valid_iter = valid_dataset

    train_loss = make_loss_compute(model, fields["tgt"].vocab, opt)
    valid_loss = make_loss_compute(model, fields["tgt"].vocab, opt)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches

    trainer = onmt.Trainer(model, train_iter, valid_iter,
                           train_loss, valid_loss, optim,
                           trunc_size, shard_size, data_type)

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validate on the validation set.
        valid_stats = trainer.validate()
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # 3. Log to remote server.
        if opt.exp_host:
            train_stats.log("train", experiment, optim.lr)
            valid_stats.log("valid", experiment, optim.lr)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats)


def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
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


def load_dataset(data_type):
    assert data_type in ["train", "valid"]

    print("Loading %s data from '%s'" % (data_type, opt.data))

    pts = glob.glob(opt.data + '.' + data_type + '.[0-9]*.pt')
    datasets = []
    if pts:
        # Multiple onmt.io.*Dataset's, coalesce all.
        # torch.load loads them imemediately, which might eat up
        # too much memory. A lazy load would be better, but later
        # when we create data iterator, it still requires these
        # data to be loaded. So it seams we don't have a good way
        # to avoid this now.
        for pt in pts:
            datasets.append(LazyIter(pt, data_type == "train"))
        # dataset = onmt.io.ONMTDatasetBase.coalesce_datasets(datasets)
    else:
        # Only one onmt.io.*Dataset, simple!
        datasets.append(LazyIter(opt.data + '.' + data_type + '.pt',
                                 data_type == "train"))

    print(' * number of %s dataset: %d' % (data_type, len(datasets)))
    return datasets


def load_fields(dataset, checkpoint):
    data_type = dataset.data_type
    fields = onmt.io.load_fields_from_vocab(
                torch.load(opt.data + '.vocab.pt'), data_type)
    fields = dict([(k, f) for (k, f) in fields.items()
                  if k in dataset.examples[0].__dict__])

    if opt.train_from:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = onmt.io.load_fields_from_vocab(
                    checkpoint['vocab'], data_type)

    if data_type == 'text':
        print(' * vocabulary size. source = %d; target = %d' %
              (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    else:
        print(' * vocabulary size. target = %d' %
              (len(fields['tgt'].vocab)))

    return fields


def collect_report_features(fields):
    src_features = onmt.io.collect_features(fields, side='src')
    tgt_features = onmt.io.collect_features(fields, side='tgt')

    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        print(' * tgt feature %d size = %d' % (j, len(fields[feat].vocab)))


def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    model = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    return model


def build_optim(model, checkpoint):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        # what members of opt does Optim need?
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)

    optim.set_parameters(model.parameters())

    return optim


def main():

    # Load train and validate data.
    train_dataset = load_dataset("train")
    valid_dataset = load_dataset("valid")
    print(' * maximum batch size: %d' % opt.batch_size)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear.
        opt.start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = opt

    # Load fields generated from preprocess phase.
    _, data = train_dataset[0].load_iterator()
    fields = load_fields(data, checkpoint)
    for wrap in train_dataset + valid_dataset:
        wrap.fields = fields

    # Report src/tgt features.
    collect_report_features(fields)

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    tally_parameters(model)
    check_save_model_path()

    # Build optimizer.
    optim = build_optim(model, checkpoint)

    # Do training.
    train_model(model, train_dataset, valid_dataset, data.data_type,
                fields, optim, model_opt)


if __name__ == "__main__":
    main()
