from __future__ import division

import os

import onmt
import onmt.Markdown
import onmt.Models
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
import dill
from train_opts import add_model_arguments, add_optim_arguments

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

# Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")

# parser.add_argument('-train_from', default='', type=str,
#                     help="""If training from a checkpoint then this is the
#                     path to the pretrained model.""")

# pretrained word vectors
parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")
parser.add_argument('-log_server', type=str, default="",
                    help="Send logs to this crayon server.")
parser.add_argument('-experiment_name', type=str, default="",
                    help="Name of the experiment for logging.")

parser.add_argument('-seed', type=int, default=-1,
                    help="""Random seed used for the experiments
                    reproducibility.""")

add_model_arguments(parser)
add_optim_arguments(parser)

opt = parser.parse_args()
print(opt)

if opt.seed > 0:
    torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)


# Set up the Crayon logging server.
if opt.log_server != "":
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=opt.log_server)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.experiment_name in experiments:
        cc.remove_experiment(opt.experiment_name)
    experiment = cc.create_experiment(opt.experiment_name)


def make_features(batch, fields):
    # This is a bit hacky for now.
    feats = []
    for j in range(100):
        key = "src_feat_" + str(j)
        if key not in fields:
            break
        feats.append(batch.__dict__[key])
    cat = [batch.src[0]] + feats
    cat = [c.unsqueeze(2) for c in cat]
    return torch.cat(cat, 2)


class LossCompute:
    def __init__(self, generator, crit, tgt_vocab, dataset, epoch):
        self.generator = generator
        self.crit = crit
        self.tgt_vocab = tgt_vocab
        self.dataset = dataset
        self.epoch = epoch

    @staticmethod
    def makeLossBatch(outputs, batch, attns, range_):
        """Create all the variables that need to be sharded.
        This needs to match compute loss exactly.
        """
        if opt.copy_attn:
            # Only copy attn uses alignment
            align = batch.alignment[range_[0] + 1: range_[1]]
        else:
            align = None
        return {"out": outputs,
                "target": batch.tgt[range_[0] + 1: range_[1]],
                "align": align,
                "coverage": attns.get("coverage"),
                "attn": attns.get("copy")}

    def computeLoss(self, batch, out, target, attn=None,
                    align=None, coverage=None):
        def bottle(v):
            return v.view(-1, v.size(2))

        def unbottle(v):
            return v.view(-1, batch.batch_size, v.size(1))

        pad = self.tgt_vocab.stoi[onmt.IO.PAD_WORD]

        target = target.view(-1)
        # print(batch.alignment[:10, 0])
        # print([self.tgt_vocab.itos[i]
        #        for i in batch.tgt.data[:10, 0]])

        if not opt.copy_attn:
            # Standard generator.
            scores = self.generator(bottle(out))
            # NLL Loss
            out = scores.gather(1, target.view(-1, 1)).view(-1)
            loss = -out.mul(target.ne(pad).float()).sum()
            scores2 = scores.data.clone()
            target = target.data.clone()
        else:
            # Copy generator. and loss.
            scores = self.generator(bottle(out), bottle(attn), batch.src_map)
            align = align.view(-1)
            offset = len(self.tgt_vocab)

            # print(scores[0])
            # print(target[0])
            # print(align[0])

            # Copy prob.
            out = scores.gather(1, align.view(-1, 1) + offset) \
                        .view(-1).mul(align.ne(0).float())
            tmp = scores.gather(1, target.view(-1, 1)).view(-1)

            # Regular prob (no unks and unks that can't be copied)
            if not opt.copy_attn_force:
                out = out + 1e-20 + tmp.mul(target.ne(0).float()) + \
                      tmp.mul(align.eq(0).float()).mul(target.eq(0).float())
            else:
                # Forced copy.
                out = out + 1e-20 + tmp.mul(align.eq(0).float())

            # print(out)
            # Drop padding.
            loss = -out.log().mul(target.ne(pad).float()).sum()

            # # Collapse scores. (No autograd, this is just for scoring)
            scores2 = scores.data.clone()
            scores2 = self.dataset.collapseCopyScores(unbottle(scores2), batch,
                                                      self.tgt_vocab)
            scores2 = bottle(scores2)

            target = target.data.clone()
            for i in range(target.size(0)):
                if target[i] == 0 and align.data[i] != 0:
                    target[i] = align.data[i] + offset

        # Coverage loss term.
        ppl = loss.data.clone()
        if opt.coverage_attn:
            cov = 0.1
            if self.epoch > 8:
                cov = 1
            loss = loss + cov * \
                torch.min(coverage, attn).sum()

        stats = onmt.Statistics.score(ppl, scores2, target, pad)
        return loss, stats


def eval(model, criterion, data, fields):
    validData = onmt.IO.OrderedIterator(
        dataset=data, device=opt.gpus[0] if opt.gpus else -1,
        batch_size=opt.batch_size, train=False, sort=True)

    stats = onmt.Statistics()
    model.eval()
    loss_compute = LossCompute(model.generator, criterion,
                               fields["tgt"].vocab, data, 0)

    for batch in validData:
        _, src_lengths = batch.src
        src = make_features(batch, fields)
        outputs, attn, _ = model(src, batch.tgt, src_lengths)
        gen_state = loss_compute.makeLossBatch(outputs, batch, attn,
                                               (0, batch.tgt.size(0)))
        _, batch_stats = loss_compute.computeLoss(batch=batch, **gen_state)
        stats.update(batch_stats)
    model.train()
    return stats


def trainModel(model, criterion, trainData, validData, fields, optim):
    model.train()

    model_dirname = os.path.dirname(opt.save_model)
    if not os.path.exists(model_dirname):
        os.mkdir(model_dirname)
    assert os.path.isdir(model_dirname), "%s not a directory" % opt.save_model

    def trainEpoch(epoch):
        model.train()
        loss_compute = LossCompute(model.generator, criterion,
                                   fields["tgt"].vocab, trainData, epoch)
        splitter = onmt.modules.Splitter(opt.max_generator_batches)

        train = onmt.IO.OrderedIterator(
            dataset=trainData, batch_size=opt.batch_size,
            device=opt.gpus[0] if opt.gpus else -1,
            repeat=False)

        total_stats = onmt.Statistics()
        report_stats = onmt.Statistics()

        # Main training loop
        for i, batch in enumerate(train):
            target_size = batch.tgt.size(0)
            dec_state = None
            _, src_lengths = batch.src
            src = make_features(batch, fields)
            report_stats.n_src_words += src_lengths.sum()

            # Truncated BPTT
            trunc_size = opt.truncated_decoder if opt.truncated_decoder \
                else target_size

            for j in range(0, target_size-1, trunc_size):

                # (1) Create truncated target.
                tgt_r = (j, j + trunc_size)
                tgt = batch.tgt[tgt_r[0]: tgt_r[1]]

                # (2) F-prop all but generator.
                model.zero_grad()
                outputs, attn, dec_state = \
                    model(src, tgt, src_lengths, dec_state)

                # (2) F-prop/B-prob generator in shards for memory
                # efficiency.
                batch_stats = onmt.Statistics()
                gen_state = loss_compute.makeLossBatch(outputs, batch, attn,
                                                       tgt_r)
                for shard in splitter.splitIter(gen_state):

                    # Compute loss and backprop shard.
                    loss, stats = loss_compute.computeLoss(batch=batch,
                                                           **shard)
                    # print("BACKWARD")
                    loss.div(batch.batch_size).backward()
                    batch_stats.update(stats)

                # (3) Update the parameters and statistics.
                optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

            # Log the statistics for the batch.
            if i % opt.log_interval == -1 % opt.log_interval:
                report_stats.output(epoch, i+1, len(train),
                                    total_stats.start_time)
                if opt.log_server:
                    report_stats.log("progress", experiment, optim)
                report_stats = onmt.Statistics()

        return total_stats

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_stats = trainEpoch(epoch)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        #  (2) evaluate on the validation set
        valid_stats = eval(model, criterion, validData, fields)
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # (optional) Log to remote server.
        if opt.log_server:
            train_stats.log("train", experiment, optim)
            valid_stats.log("valid", experiment, optim)

        #  (3) update the learning rate
        optim.updateLearningRate(valid_stats.ppl(), epoch)

        #  (4) drop a checkpoint
        if epoch >= opt.start_checkpoint_at:
            model_state_dict = (model.module.state_dict()
                                if len(opt.gpus) > 1
                                else model.state_dict())
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if 'generator' not in k}
            generator_state_dict = (model.generator.module.state_dict()
                                    if len(opt.gpus) > 1
                                    else model.generator.state_dict())

            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'fields': fields,
                'opt': opt,
                'epoch': epoch,
                'optim': optim
            }
            torch.save(checkpoint,
                       '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                       % (opt.save_model, valid_stats.accuracy(),
                          valid_stats.ppl(), epoch), pickle_module=dill)


def main():
    train = torch.load(opt.data + '.train.pt', pickle_module=dill)
    fields = torch.load(opt.data + '.fields.pt', pickle_module=dill)
    valid = torch.load(opt.data + '.valid.pt', pickle_module=dill)
    fields = dict(fields)
    src_features = [fields["src_feat_"+str(j)]
                    for j in range(train.nfeatures)]

    checkpoint = None
    dict_checkpoint = opt.train_from_state_dict
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        fields = checkpoint['fields']

    print(' * vocabulary size. source = %d; target = %d' %
          (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' %
              (j, len(feat.vocab)))
    print(' * number of training sentences. %d' %
          len(train))
    print(' * maximum batch size. %d' % opt.batch_size)
    print('Building model...')

    cuda = (len(opt.gpus) >= 1)
    model = onmt.Models.make_base_model(opt, opt, fields, cuda, checkpoint)
    print(model)

    # Define criterion of each GPU.
    # Multi-gpu
    if len(opt.gpus) > 1:
        print('Multi gpu training ', opt.gpus)
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        model.generator = nn.DataParallel(model.generator, device_ids=opt.gpus,
                                          dim=0)

    if not opt.train_from_state_dict:
        # Param initialization.
        if opt.param_init != 0.0:
            print('Intializing params')
            for p in model.parameters():
                p.data.uniform_(-opt.param_init, opt.param_init)

        model.encoder.embeddings.load_pretrained_vectors(opt.pre_word_vecs_enc)
        model.decoder.embeddings.load_pretrained_vectors(opt.pre_word_vecs_dec)
        optim = onmt.Optim(opt)
        optim.set_parameters(model.parameters())
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        optim.set_parameters(model.parameters())
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
        else:
            print(name, param.nelement())
    print('encoder: ', enc)
    print('decoder: ', dec)

    trainModel(model, None, train, valid, fields, optim)


if __name__ == "__main__":
    main()
