from __future__ import division

import onmt
import onmt.Markdown
import onmt.Models
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
import torchtext.data

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

# Data options

# parser.add_argument('-data', required=True,
#                     help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

# Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=500,
                    help='Word embedding sizes')
parser.add_argument('-feature_vec_size', type=int, default=100,
                    help='Feature vec sizes')

parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
parser.add_argument('-rnn_type', type=str, default='LSTM',
                    choices=['LSTM', 'GRU'],
                    help="""The gate type to use in the RNNs""")
# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")
parser.add_argument('-copy_attn', action="store_true",
                    help='Train copy attention layer.')
parser.add_argument('-coverage_attn', action="store_true",
                    help='Train a coverage attention layer.')
parser.add_argument('-lambda_coverage', type=float, default=1,
                    help='Lambda value for coverage.')

parser.add_argument('-encoder_layer', type=str, default='rnn',
                    help="""Type of encoder layer to use.
                    Options: [rnn|mean|transformer]""")
parser.add_argument('-decoder_layer', type=str, default='rnn',
                    help='Type of decoder layer to use. [rnn|transformer]')
parser.add_argument('-context_gate', type=str, default=None,
                    choices=['source', 'target', 'both'],
                    help="""Type of context gate to use [source|target|both].
                    Do not select for no context gate.""")
parser.add_argument('-attention_type', type=str, default='dotprod',
                    choices=['dotprod', 'mlp'],
                    help="""The attention type to use:
                    dotprot (Luong) or MLP (Bahdanau)""")

# Optimization options
parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img].")
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init).
                    Use 0 to not use initialization""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-position_encoding', action='store_true',
                    help='Use a sinusoid to mark relative words positions.')
parser.add_argument('-share_decoder_embeddings', action='store_true',
                    help='Share the word and softmax embeddings for decoder.')

parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")
parser.add_argument('-truncated_decoder', type=int, default=0,
                    help="""Truncated bptt.""")

# learning rate
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1,
                    adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=8,
                    help="""Start decaying every epoch after and including this
                    epoch""")
parser.add_argument('-start_checkpoint_at', type=int, default=0,
                    help="""Start checkpointing every epoch after and including this
                    epoch""")
parser.add_argument('-decay_method', type=str, default="",
                    help="""Use a custom learning rate decay [|noam] """)
parser.add_argument('-warmup_steps', type=int, default=4000,
                    help="""Number of warmup steps for custom decay.""")


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


parser.add_argument('-src_type', default="text",
                    help="Type of the source input. Options are [text|img].")
parser.add_argument('-src_img_dir', default=".",
                    help="Location of source images")

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                    help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")
parser.add_argument('-features_vocabs_prefix', type=str, default='',
                    help="Path prefix to existing features vocabularies")
parser.add_argument('-src_seq_length', type=int, default=50,
                    help="Maximum source sequence length")
parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                    help="Truncate source sequence length.")
parser.add_argument('-tgt_seq_length', type=int, default=50,
                    help="Maximum target sequence length to keep.")
parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                    help="Truncate target sequence length.")

parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
# parser.add_argument('-seed',       type=int, default=3435,
#                     help="Random seed")



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


def eval(model, criterion, data):
    stats = onmt.Statistics()
    model.eval()
    loss_compute = LossCompute(model.generator, criterion)

    for i in range(len(data)):
        batch = data[i]
        outputs, attn, _ = model(batch.src[0], batch.tgt, batch.src[1])
        gen_state = loss_compute.makeLossBatch(outputs, batch, attn)
        _, batch_stats = loss_compute.computeLoss(**gen_state)
        stats.update(batch_stats)
    model.train()
    return stats


class LossCompute:
    def __init__(self, generator, crit):
        self.generator = generator
        self.crit = crit

    @staticmethod
    def makeLossBatch(outputs, batch, attns, range_):
        return {"out": outputs,
                "target": batch.tgt[range_[0] + 1: range_[1]],
                "align": batch.alignment[range_[0] + 1: range_[1]],
                "coverage": attns.get("coverage"),
                "attn": attns.get("copy")}

    def computeLoss(self, out, target, attn=None, align=None, coverage=None):
        def bottle(v):
            return v.view(-1, v.size(2))

        if not opt.copy_attn:
            # Standard loss.
            scores = self.generator(bottle(out))
            loss = self.crit(scores, target.view(-1))
        else:
            # Need extra args for copy.
            scores, c_attn = self.generator(bottle(out), bottle(attn))
            loss = self.crit(scores, c_attn, target, bottle(align))

        # Coverage can be applied for either.
        if opt.coverage_attn:
            loss += opt.lambda_coverage * \
                    torch.min(coverage, attn).sum()

        stats = onmt.Statistics.score(loss.data, scores.data, target.data)
        return loss, stats


def trainModel(model, criterion, trainData, validData, dataset, optim):
    model.train()
    loss_compute = LossCompute(model.generator, criterion)
    splitter = onmt.modules.Splitter(opt.max_generator_batches)

    def trainEpoch(epoch):
        # if opt.extra_shuffle and epoch > opt.curriculum:
        #     trainData.shuffle()

        # # Shuffle mini batch order.
        # batchOrder = torch.randperm(len(trainData))
        # for i in range(len(trainData)):

        total_stats = onmt.Statistics()
        report_stats = onmt.Statistics()

        for i, batch in enumerate(trainData):
            target_size = batch.tgt.size(0)
            dec_state = None
            trunc_size = opt.truncated_decoder if opt.truncated_decoder \
                else target_size

            for j in range(0, target_size-1, trunc_size):
                # Main training loop
                src, src_lengths = batch.src
                src = src.unsqueeze(2)

                model.zero_grad()
                outputs, attn, dec_state = \
                    model(src, batch.tgt[j: j + trunc_size], src_lengths, dec_state)

                gen_state = loss_compute.makeLossBatch(outputs, batch, attn,
                                                       (j, j + trunc_size))
                batch_stats = onmt.Statistics()
                for shard in splitter.splitIter(gen_state):
                    loss, stats = loss_compute.computeLoss(**shard)

                    # Compute statistics.
                    batch_stats.update(stats)
                    loss.div(batch.batch_size).backward()

                # Update the parameters.
                optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

            report_stats.n_src_words += src_lengths.sum()

            if i % opt.log_interval == -1 % opt.log_interval:
                report_stats.output(epoch, i+1, len(trainData),
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
        valid_stats = eval(model, criterion, validData)
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # Log to remote server.
        if opt.log_server:
            train_stats.log("train", experiment, optim)
            valid_stats.log("valid", experiment, optim)

        #  (3) update the learning rate
        optim.updateLearningRate(valid_stats.ppl(), epoch)

        model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                            else model.state_dict())
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = (model.generator.module.state_dict()
                                if len(opt.gpus) > 1
                                else model.generator.state_dict())
        #  (4) drop a checkpoint
        if epoch >= opt.start_checkpoint_at:
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'dicts': dataset['dicts'],
                'opt': opt,
                'epoch': epoch,
                'optim': optim
            }
            torch.save(checkpoint,
                       '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                       % (opt.save_model, valid_stats.accuracy(),
                          valid_stats.ppl(), epoch))


def main():
    fields = onmt.IO.ONMTDataset.get_fields(opt.train_src, opt.train_tgt)
    train = onmt.IO.ONMTDataset(opt.train_src, opt.train_tgt, fields, opt)
    vocabs = onmt.IO.ONMTDataset.build_vocab(train, fields, opt)
    valid = onmt.IO.ONMTDataset(opt.valid_src, opt.valid_tgt, fields, opt)
    fields = dict(fields)

    dict_checkpoint = (opt.train_from if opt.train_from
                       else opt.train_from_state_dict)
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    trainData = onmt.IO.OrderedIterator(
        dataset=train, batch_size=opt.batch_size, sort=True,
        device=opt.gpus if opt.gpus else -1)

    validData = onmt.IO.OrderedIterator(
        dataset=valid, device=opt.gpus if opt.gpus else -1,
        batch_size=opt.batch_size, train=False, sort=True)

    print(' * vocabulary size. source = %d; target = %d' %
          (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    # if 'src_features' in dicts:
    #     for j in range(len(dicts['src_features'])):
    #         print(' * src feature %d size = %d' %
    #               (j, ['src_features'][j].size()))

    print(' * number of training sentences. %d' %
          len(train))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    if opt.encoder_type == "text":
        encoder = onmt.Models.Encoder(opt, fields['src'].vocab,
                                      fields.get('src_features', None))
    elif opt.encoder_type == "img":
        encoder = onmt.modules.ImageEncoder(opt)
        assert("type" not in dataset or dataset["type"] == "img")
    else:
        print("Unsupported encoder type %s" % (opt.encoder_type))

    decoder = onmt.Models.Decoder(opt, fields['tgt'].vocab)

    if opt.copy_attn:
        generator = onmt.modules.CopyGenerator(opt, dicts['src'], dicts['tgt'])
    else:
        generator = nn.Sequential(
            nn.Linear(opt.rnn_size, len(fields['tgt'].vocab)),
            nn.LogSoftmax())
        if opt.share_decoder_embeddings:
            generator[0].weight = decoder.word_lut.weight

    model = onmt.Models.NMTModel(encoder, decoder, len(opt.gpus) > 1)

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        chk_model = checkpoint['model']
        generator_state_dict = chk_model.generator.state_dict()
        model_state_dict = {k: v for k, v in chk_model.state_dict().items()
                            if 'generator' not in k}
        model.load_state_dict(model_state_dict)
        generator.load_state_dict(generator_state_dict)
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_from_state_dict:
        print('Loading model from checkpoint at %s'
              % opt.train_from_state_dict)
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
        opt.start_epoch = checkpoint['epoch'] + 1

    # Define criterion of each GPU.
    vocabSize = len(fields['tgt'].vocab)
    if not opt.copy_attn:
        weight = torch.ones(vocabSize)
        weight[onmt.Constants.PAD] = 0
        criterion = nn.NLLLoss(weight, size_average=False)
    else:
        criterion = onmt.modules.CopyCriterion

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
        criterion.cuda()
    else:
        model.cpu()
        generator.cpu()
        criterion.cpu()

    if len(opt.gpus) > 1:
        print('Multi gpu training ', opt.gpus)
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

    model.generator = generator

    if not opt.train_from_state_dict and not opt.train_from:
        if opt.param_init != 0.0:
            print('Intializing params')
            for p in model.parameters():
                p.data.uniform_(-opt.param_init, opt.param_init)

        encoder.embeddings.load_pretrained_vectors(opt.pre_word_vecs_enc)
        decoder.embeddings.load_pretrained_vectors(opt.pre_word_vecs_dec)

        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            opt=opt
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)

    optim.set_parameters(model.parameters())

    if opt.train_from or opt.train_from_state_dict:
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    # trainModel(model, criterion, trainData, validData, dataset, optim)
    trainModel(model, criterion, trainData, validData, None, optim)

if __name__ == "__main__":
    main()
