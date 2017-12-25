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

import numpy as np
from tqdm import tqdm
from gleu import GLEU


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


def make_train_data_iter(train_dataset, opt):
    """
    This returns user-defined train data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    return onmt.io.OrderedIterator(
                dataset=train_dataset, batch_size=opt.batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                repeat=False)


def make_valid_data_iter(valid_dataset, opt):
    """
    This returns user-defined validate data iterator for the trainer
    to iterate over during each validate epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    is ok too.
    """
    return onmt.io.OrderedIterator(
                dataset=valid_dataset, batch_size=opt.batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                train=False, sort=True)


def make_loss_compute(model, tgt_vocab, dataset, opt):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, dataset, opt.copy_attn_force)
    else:
        compute = onmt.Loss.NMTLossCompute(model.generator, tgt_vocab,
                                           opt.label_smoothing)

    if use_gpu(opt):
        compute.cuda()

    return compute


def compute_gleu_score(src, n_ref, pred):

    def avg_gleu_score(scores):
        scores = [float(l[0]) for l in scores if l!='nan']
        return np.average(scores)

    gleu_calculator = GLEU(len(n_ref))
    gleu_calculator.load_sources(src)
    gleu_calculator.load_references(n_ref)
    scores = [g for g in gleu_calculator.run_iterations(n=len(n_ref),
                                                        hypothesis=pred, 
                                                        per_sent=False)]
    return avg_gleu_score(scores)


def validate_gleu_score(model, translator, translation_builder, val_dataset, val_dataset_iter, epoch, validation_dump_dir):

    assert opt.n_best==1, "Current GLEU calculation only support single hypothesis, got {}".format(opt.n_best)

    all_translations = []

    # Do Translation
    #
    # Note: 
    #       For simplicity, we currently do translation for each reference, in other word, we do extra 3 redundant translations.
    #       Will comeback to fix this if necessary
    model.eval()
    model.generator.eval()

    for batch in tqdm(val_dataset_iter):
        batch_data = translator.translate_batch(batch, val_dataset)
        translations = translation_builder.from_batch(batch_data)
        all_translations += translations

    model.train()
    model.generator.train()
    assert len(all_translations)%4==0, "Expected source should be 4's multiplier, got {}".format(len(all_translations)) 

    # Transform list to dict, using src as dict key

    all_translations_dict = {}
    for translation in all_translations:
        src = translation.src_raw
        key = tuple(src)
        if key in all_translations_dict:
            ref = translation.gold_sent
            all_translations_dict[key]['refs'].append(ref)
        else:
            obj = {}
            all_translations_dict[key] = obj
            pred = translation.pred_sents[0]
            ref = translation.gold_sent
            obj['src'] = src
            obj['pred'] = pred
            obj['refs'] = [ref]
        
    # Collapse dict back to list

    all_src = []
    all_pred = []
    all_refs = [[] for i in range(4)]
    for k in all_translations_dict:
        all_src.append(all_translations_dict[k]['src'])
        all_pred.append(all_translations_dict[k]['pred'])
        for n in range(4):
            all_refs[n].append(all_translations_dict[k]['refs'][n])

    avg_gleu_score = compute_gleu_score(all_src, all_refs, all_pred)

    # Save validation output for each epoch. This can help analysis and debugging.
    
    with open(validation_dump_dir + 'Epoch_{}.txt'.format(epoch), 'w') as f:
        for i in range(len(all_src)):
            f.write(str(i)+'\n')
            f.write(' '.join(all_src[i])+'\n')
            f.write(' '.join(all_pred[i])+'\n')
            for n in range(4):
                f.write(' '.join(all_refs[n][i])+'\n')


    return avg_gleu_score


def train_model(model, train_dataset, valid_dataset,
                fields, optim, model_opt):

    train_iter = make_train_data_iter(train_dataset, opt)
    valid_iter = make_valid_data_iter(valid_dataset, opt)

    train_loss = make_loss_compute(model, fields["tgt"].vocab,
                                   train_dataset, opt)
    valid_loss = make_loss_compute(model, fields["tgt"].vocab,
                                   valid_dataset, opt)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    data_type = train_dataset.data_type

    # Setup translator for GLEU score 
    # Since OpenNMT model has copy mechenism, 
    #   we should fully reconstruct output instead of using model output to get GLEU score directly
    if opt.eval_metric == 'gleu':

        # Translator
        val_dataset = onmt.io.build_dataset(fields, "text",
                                            opt.gleu_src_path, opt.gleu_tgt_path,
                                            src_dir=os.path.dirname(opt.gleu_src_path),
                                            sample_rate=opt.sample_rate,
                                            window_size=opt.window_size,
                                            window_stride=opt.window_stride,
                                            window=opt.window,
                                            use_filter_pred=False)

        val_dataset_iter = onmt.io.OrderedIterator(dataset=val_dataset, device=opt.gpuid[0],
                                                batch_size=opt.batch_size, train=False, sort=False,
                                                shuffle=False)

        scorer = onmt.translate.GNMTGlobalScorer(opt.alpha, opt.beta)

        translator = onmt.translate.Translator(model, fields,
                                               beam_size=opt.beam_size,
                                               n_best=opt.n_best,
                                               global_scorer=scorer,
                                               max_length=opt.max_sent_length,
                                               copy_attn=model_opt.copy_attn,
                                               cuda=True,
                                               beam_trace=opt.dump_beam != "")

        translation_builder = onmt.translate.TranslationBuilder(val_dataset, translator.fields,
                                                                opt.n_best, opt.replace_unk, opt.gleu_tgt_path)

        import time

        localtime   = time.localtime()
        validation_dump_dir = './validation_dump/val_{}/'.format(time.strftime("%Y:%m:%d_%H:%M:%S", time.localtime()))
        os.mkdir(validation_dump_dir)

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

        # 2-1. Validate GLEU score
        if opt.eval_metric == 'gleu':

            model.eval()
            model.generator.eval()
            
            val_gleu_avg_score = validate_gleu_score(model, translator, translation_builder, val_dataset, val_dataset_iter, epoch, validation_dump_dir)
            
            model.train()
            model.generator.train()

            print('Validation GLEU avg score: %g' % val_gleu_avg_score)

        # 3. Log to remote server.
        if opt.exp_host:
            train_stats.log("train", experiment, optim.lr)
            valid_stats.log("valid", experiment, optim.lr)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            if opt.eval_metric == 'gleu':
                trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats, val_gleu_avg_score=val_gleu_avg_score)
            else:
                trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats, val_gleu_avg_score=None)


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
    if pts:
        # Multiple onmt.io.*Dataset's, coalesce all.
        # torch.load loads them imemediately, which might eat up
        # too much memory. A lazy load would be better, but later
        # when we create data iterator, it still requires these
        # data to be loaded. So it seams we don't have a good way
        # to avoid this now.
        datasets = []
        for pt in pts:
            datasets.append(torch.load(pt))
        dataset = onmt.io.ONMTDatasetBase.coalesce_datasets(datasets)
    else:
        # Only one onmt.io.*Dataset, simple!
        dataset = torch.load(opt.data + '.' + data_type + '.pt')

    print(' * number of %s sentences: %d' % (data_type, len(dataset)))

    return dataset


def load_fields(train_dataset, valid_dataset, checkpoint):
    data_type = train_dataset.data_type

    fields = onmt.io.load_fields_from_vocab(
                torch.load(opt.data + '.vocab.pt'), data_type)
    fields = dict([(k, f) for (k, f) in fields.items()
                  if k in train_dataset.examples[0].__dict__])

    # We save fields in vocab.pt, so assign them back to dataset here.
    train_dataset.fields = fields
    valid_dataset.fields = fields

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
            opt=opt
        )

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
    fields = load_fields(train_dataset, valid_dataset, checkpoint)

    # Report src/tgt features.
    collect_report_features(fields)

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    tally_parameters(model)
    check_save_model_path()

    # Build optimizer.
    optim = build_optim(model, checkpoint)

    # Do training.
    train_model(model, train_dataset, valid_dataset, fields, optim, model_opt)


if __name__ == "__main__":
    main()
