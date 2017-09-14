from __future__ import division

import os
import argparse
import torch
import torch.nn as nn
from torch import cuda

import onmt
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu
import opts

parser = argparse.ArgumentParser(description='train.py')

# Data and loading options
parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')

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
    torch.manual_seed(opt.seed)

if opt.rnn_type == "SRU" and not opt.gpuid:
    raise AssertionError("Using SRU requires -gpuid set.")

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)


# Set up the Crayon logging server.
if opt.exp_host != "":
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=opt.exp_host)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.exp in experiments:
        cc.remove_experiment(opt.exp)
    experiment = cc.create_experiment(opt.exp)


def eval(model, criterion, data, fields):
    valid_data = onmt.IO.OrderedIterator(
        dataset=data, device=opt.gpuid[0] if opt.gpuid else -1,
        batch_size=opt.batch_size, train=False, sort=True)

    stats = onmt.Loss.Statistics()
    model.eval()
    loss = onmt.Loss.LossCompute(model.generator, criterion,
                                 fields["tgt"].vocab, data, 0, opt.copy_attn)
    for batch in valid_data:
        _, src_lengths = batch.src
        src = onmt.IO.make_features(batch, 'src')
        tgt = onmt.IO.make_features(batch, 'tgt')
        outputs, attn, _ = model(src, tgt, src_lengths)
        gen_state = loss.make_loss_batch(
            outputs, batch, attn, (0, batch.tgt.size(0)))
        _, batch_stats = loss.compute_loss(batch=batch, **gen_state)
        stats.update(batch_stats)
    model.train()
    return stats


def train_model(model, train_data, valid_data, fields, optim):
    model.train()

    padding_idx = fields['tgt'].vocab.stoi[onmt.IO.PAD_WORD]

    # Define criterion of each GPU.
    if not opt.copy_attn:
        criterion = onmt.Loss.nmt_criterion(
            len(fields['tgt'].vocab), opt.gpuid, padding_idx)
    else:
        criterion = onmt.modules.CopyCriterion(len(fields['tgt'].vocab),
                                               opt.copy_attn_force,
                                               padding_idx)

    train = onmt.IO.OrderedIterator(
        dataset=train_data, batch_size=opt.batch_size,
        device=opt.gpuid[0] if opt.gpuid else -1,
        repeat=False)

    def train_epoch(epoch):
        closs = onmt.Loss.LossCompute(model.generator, criterion,
                                      fields["tgt"].vocab, train_data,
                                      epoch, opt.copy_attn)

        total_stats = onmt.Loss.Statistics()
        report_stats = onmt.Loss.Statistics()

        for i, batch in enumerate(train):
            target_size = batch.tgt.size(0)

            dec_state = None
            _, src_lengths = batch.src

            src = onmt.IO.make_features(batch, 'src')
            tgt = onmt.IO.make_features(batch, 'tgt')
            report_stats.n_src_words += src_lengths.sum()

            # Truncated BPTT
            trunc_size = opt.truncated_decoder if opt.truncated_decoder \
                else target_size

            for j in range(0, target_size-1, trunc_size):
                # (1) Create truncated target.
                tgt = tgt[j: j + trunc_size]

                # (2) F-prop all but generator.

                # Main training loop
                model.zero_grad()
                outputs, attn, dec_state = \
                    model(src, tgt, src_lengths, dec_state)

                # (2) F-prop/B-prob generator in shards for memory
                # efficiency.
                batch_stats = onmt.Loss.Statistics()
                # make_loss_batch doesn't really need to be a method of
                # ComputeLoss
                gen_state = closs.make_loss_batch(outputs, batch, attn,
                                                  (j, j + trunc_size))
                shard_size = opt.max_generator_batches
                for shard in onmt.Loss.shards(gen_state, shard_size):

                    # Compute loss and backprop shard.
                    loss, stats = closs.compute_loss(batch=batch,
                                                     **shard)
                    loss.div(batch.batch_size).backward()
                    batch_stats.update(stats)

                # (3) Update the parameters and statistics.
                optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

            if i % opt.report_every == -1 % opt.report_every:
                report_stats.output(epoch, i+1, len(train),
                                    total_stats.start_time)
                if opt.exp_host:
                    report_stats.log("progress", experiment, optim)
                report_stats = onmt.Loss.Statistics()
        return total_stats

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print()

        #  (1) train for one epoch on the training set
        train_stats = train_epoch(epoch)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        #  (2) evaluate on the validation set
        valid_stats = eval(model, criterion, valid_data, fields)
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # Log to remote server.
        if opt.exp_host:
            train_stats.log("train", experiment, optim)
            valid_stats.log("valid", experiment, optim)

        #  (3) update the learning rate
        optim.updateLearningRate(valid_stats.ppl(), epoch)

        model_state_dict = (model.module.state_dict() if len(opt.gpuid) > 1
                            else model.state_dict())
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = (model.generator.module.state_dict()
                                if len(opt.gpuid) > 1
                                else model.generator.state_dict())
        #  (4) drop a checkpoint
        if epoch >= opt.start_checkpoint_at:
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'vocab': onmt.IO.ONMTDataset.save_vocab(fields),
                'opt': opt,
                'epoch': epoch,
                'optim': optim
            }
            torch.save(checkpoint,
                       '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                       % (opt.save_model, valid_stats.accuracy(),
                          valid_stats.ppl(), epoch))


def check_model_path():
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


def main():
    print("Loading data from '%s'" % opt.data)

    train = torch.load(opt.data + '.train.pt')
    valid = torch.load(opt.data + '.valid.pt')

    fields = onmt.IO.ONMTDataset.load_fields(
        torch.load(opt.data + '.vocab.pt'))
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in train.examples[0].__dict__])

    train.fields = fields
    valid.fields = fields
    # TODO: account for target features. Also, why does fields need to
    # have the structure it does?
    src_features = [fields["src_feat_"+str(j)]
                    for j in range(train.nfeatures)]
    model_opt = opt
    checkpoint = None

    if opt.train_from:
        print('Loading dicts from checkpoint at %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        fields = onmt.IO.ONMTDataset.load_fields(checkpoint['vocab'])
        model_opt = checkpoint["opt"]

    print(' * vocabulary size. source = %d; target = %d' %
          (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(feat.vocab)))

    print(' * number of training sentences. %d' % len(train))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    model = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    # Load model from checkpoint or initialize, create optim
    if opt.train_from:
        print('Loading model from checkpoint at %s'
              % opt.train_from)
        # I don't like reassigning attributes of opt: it's not clear
        opt.start_epoch = checkpoint['epoch'] + 1

        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        if opt.param_init != 0.0:
            print('Intializing params')
            for p in model.parameters():
                p.data.uniform_(-opt.param_init, opt.param_init)
        model.encoder.embeddings.load_pretrained_vectors(opt.pre_word_vecs_enc,
                                                         opt.fix_word_vecs_enc)
        model.decoder.embeddings.load_pretrained_vectors(opt.pre_word_vecs_dec,
                                                         opt.fix_word_vecs_dec)
        # what members of opt does Optim need?
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            opt=opt
        )

    optim.set_parameters(model.parameters())

    tally_parameters(model)

    check_model_path()

    train_model(model, train, valid, fields, optim)


if __name__ == "__main__":
    main()
