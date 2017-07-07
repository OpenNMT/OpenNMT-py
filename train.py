from __future__ import division

import onmt
import onmt.Markdown
import onmt.Models
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda

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
    stats = onmt.Loss.Statistics()
    model.eval()
    loss = onmt.Loss.MemoryEfficientLoss(opt, model.generator, criterion,
                                         eval=True, copy_loss=opt.copy_attn)
    for i in range(len(data)):
        batch = data[i]
        outputs, attn, dec_hidden = model(batch.src, batch.tgt, batch.lengths)
        batch_stats, _, _ = loss.loss(batch, outputs, attn)
        stats.update(batch_stats)
    model.train()
    return stats


def trainModel(model, trainData, validData, dataset, optim):
    model.train()

    # Define criterion of each GPU.
    if not opt.copy_attn:
        criterion = onmt.Loss.NMTCriterion(dataset['dicts']['tgt'].size(), opt)
    else:
        criterion = onmt.modules.CopyCriterion

    def trainEpoch(epoch):
        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        mem_loss = onmt.Loss.MemoryEfficientLoss(opt, model.generator,
                                                 criterion,
                                                 copy_loss=opt.copy_attn)

        # Shuffle mini batch order.
        batchOrder = torch.randperm(len(trainData))

        total_stats = onmt.Loss.Statistics()
        report_stats = onmt.Loss.Statistics()

        for i in range(len(trainData)):
            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx]
            target_size = batch.tgt.size(0)

            dec_state = None
            trunc_size = opt.truncated_decoder if opt.truncated_decoder \
                else target_size

            for j in range(0, target_size-1, trunc_size):
                trunc_batch = batch.truncate(j, j + trunc_size)

                # Main training loop
                model.zero_grad()
                outputs, attn, dec_state = model(trunc_batch.src,
                                                 trunc_batch.tgt,
                                                 trunc_batch.lengths,
                                                 dec_state)
                batch_stats, inputs, grads \
                    = mem_loss.loss(trunc_batch, outputs, attn)

                torch.autograd.backward(inputs, grads)

                # Update the parameters.
                optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)
                if dec_state is not None:
                    dec_state.detach()

            report_stats.n_src_words += batch.lengths.data.sum()

            if i % opt.log_interval == -1 % opt.log_interval:
                report_stats.output(epoch, i+1, len(trainData),
                                    total_stats.start_time)
                if opt.log_server:
                    report_stats.log("progress", experiment, optim)
                report_stats = onmt.Loss.Statistics()

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
            train_stats.log("train", optim, experiment)
            valid_stats.log("valid", optim, experiment)

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
    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)
    dict_checkpoint = (opt.train_from if opt.train_from
                       else opt.train_from_state_dict)
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.gpus,
                             data_type=dataset.get("type", "text"),
                             srcFeatures=dataset['train'].get('src_features'),
                             tgtFeatures=dataset['train'].get('tgt_features'),
                             alignment=dataset['train'].get('alignments'))
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.gpus,
                             volatile=True,
                             data_type=dataset.get("type", "text"),
                             srcFeatures=dataset['valid'].get('src_features'),
                             tgtFeatures=dataset['valid'].get('tgt_features'),
                             alignment=dataset['valid'].get('alignments'))

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    if 'src_features' in dicts:
        for j in range(len(dicts['src_features'])):
            print(' * src feature %d size = %d' %
                  (j, dicts['src_features'][j].size()))

    dicts = dataset['dicts']
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    if opt.encoder_type == "text":
        encoder = onmt.Models.Encoder(opt, dicts['src'],
                                      dicts.get('src_features', None))
    elif opt.encoder_type == "img":
        encoder = onmt.modules.ImageEncoder(opt)
        assert("type" not in dataset or dataset["type"] == "img")
    else:
        print("Unsupported encoder type %s" % (opt.encoder_type))

    decoder = onmt.Models.Decoder(opt, dicts['tgt'])

    if opt.copy_attn:
        generator = onmt.modules.CopyGenerator(opt, dicts['src'], dicts['tgt'])
    else:
        generator = nn.Sequential(
            nn.Linear(opt.rnn_size, dicts['tgt'].size()),
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

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()

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

    trainModel(model, trainData, validData, dataset, optim)


if __name__ == "__main__":
    main()
