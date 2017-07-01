from __future__ import division

import onmt
import onmt.Markdown
import onmt.Models
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import sys
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
parser.add_argument('-encoder_layer', type=str, default='rnn',
                    help="""Type of encoder layer to use.
                    Options: [rnn|mean|transformer]""")
parser.add_argument('-decoder_layer', type=str, default='rnn',
                    help='Type of decoder layer to use. [rnn|transformer]')
parser.add_argument('-context_gate', type=str, default=None,
                    choices=['source', 'target', 'both'],
                    help="""Type of context gate to use [source|target|both].
                    Do not select for no context gate.""")

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

if opt.log_server != "":
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=opt.log_server)
    cc.remove_experiment(opt.experiment_name)
    experiment = cc.create_experiment(opt.experiment_name)


def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


def memoryEfficientLoss(outputs, generator, crit, batch,
                        eval=False,
                        attns=None, coverage=None, copy=None):
    """
    Args:
        outputs (FloatTensor): tgt_len x batch x rnn_size
        generator (Function): ( any x rnn_size ) -> ( any x tgt_vocab )
        crit (Criterion): ( any x tgt_vocab )
        batch (`Batch`): Data object
        eval (bool): train or eval
        attns (FloatTensor): src_len x batch

    Returns:
        loss (float): accumulated loss value
        grad_output: grad of loss wrt outputs
        grad_attns: grad of loss wrt attns
        num_correct (int): number of correct targets

    """
    targets = batch.tgt[1:]

    # compute generations one piece at a time
    num_correct, loss = 0, 0

    # These will require gradients.
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)
    batch_size = batch.batchSize
    d = {"out": outputs, "tgt": targets}

    if attns is not None:
        attns = Variable(attns.data, requires_grad=(not eval), volatile=eval)
        d["attn"] = attns
        d["align"] = batch.alignment[1:]

    if coverage is not None:
        coverage = Variable(coverage.data, requires_grad=(not eval),
                            volatile=eval)
        d["coverage"] = coverage

    for k in d:
        d[k] = torch.split(d[k], opt.max_generator_batches)

    for i, targ_t in enumerate(d["tgt"]):
        out_t = d["out"][i].view(-1, d["out"][i].size(2))

        # Depending on generator type.
        if attns is None:
            scores_t = generator(out_t)
            loss_t = crit(scores_t, targ_t.view(-1))
        else:
            attn_t = d["attn"][i]
            align_t = d["align"][i].view(-1, d["align"][i].size(2))
            words = batch.words().t().contiguous()
            attn_t = attn_t.view(-1, d["attn"][i].size(2))

            # probability of words, probability of attn
            scores_t, c_attn_t = generator(out_t, words, attn_t)
            loss_t = crit(scores_t, c_attn_t, targ_t.view(-1), align_t)

        if coverage is not None:
            loss_t += 0.1 * torch.min(d["coverage"][i], d["attn"][i]).sum()

        pred_t = scores_t.data.max(1)[1]
        num_correct_t = pred_t.eq(targ_t.data) \
                              .masked_select(
                                  targ_t.ne(onmt.Constants.PAD).data) \
                              .sum()
        num_correct += num_correct_t
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()

    # Return the gradients
    grad_output = None if outputs.grad is None else outputs.grad.data
    grad_attns = None if not attns or attns.grad is None else attns.grad.data
    grad_coverage = None if not coverage or coverage.grad is None \
        else coverage.grad.data

    return loss, grad_output, grad_attns, grad_coverage, num_correct


def eval(model, criterion, data):
    total_loss = 0
    total_words = 0
    total_num_correct = 0
    model.eval()
    for i in range(len(data)):
        batch = data[i]
        outputs, attn, dec_hidden = model(batch)
        # exclude <s> from targets
        targets = batch.tgt[1:]
        loss, _, _, _, num_correct = memoryEfficientLoss(
            outputs, model.generator, criterion, batch, eval=True,
            attns=attn.get("copy"), coverage=attn.get("coverage"))
        total_loss += loss
        total_num_correct += num_correct
        total_words += targets.data.ne(onmt.Constants.PAD).sum()

    model.train()
    return total_loss / total_words, total_num_correct / total_words


def trainModel(model, trainData, validData, dataset, optim):
    # print(model)
    model.train()

    # Define criterion of each GPU.
    if not opt.copy_attn:
        criterion = NMTCriterion(dataset['dicts']['tgt'].size())
    else:
        criterion = onmt.modules.copy_criterion

    start_time = time.time()

    def trainEpoch(epoch):

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # Shuffle mini batch order.
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_tgt_words = 0, 0
        report_src_words, report_num_correct = 0, 0
        start = time.time()
        for i in range(len(trainData)):

            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx]

            dec_hidden = None

            trunc_size = opt.truncated_decoder
            r = [0]
            if trunc_size:
                r = range((batch.tgt.size(0) // trunc_size) + 1)
            for j in r:
                if trunc_size:
                    if batch.tgt.size(0) - 1 <= j * trunc_size:
                        continue
                    trunc_batch = batch.truncate(j * trunc_size,
                                                 (j+1) * trunc_size)
                else:
                    trunc_batch = batch

                # Main training loop
                model.zero_grad()
                outputs, attn, dec_hidden \
                    = model(trunc_batch,
                            dec_hidden=(h.detach()for h in dec_hidden)
                            if dec_hidden else None)

                # Exclude <s> from targets.
                targets = trunc_batch.tgt[1:]
                loss, gradOutput, gradAttn, gradCov, num_correct \
                    = memoryEfficientLoss(
                        outputs, model.generator, criterion, trunc_batch,
                        attns=attn.get("copy"), coverage=attn.get("coverage"))
                var, grad = [outputs], [gradOutput]
                if gradAttn is not None:
                    var, grad = [outputs, attn["copy"]], [gradOutput, gradAttn]
                if gradCov is not None:
                    var.append(attn["coverage"])
                    grad.append(gradCov)
                torch.autograd.backward(var, grad)
                # Update the parameters.
                optim.step()

                num_words = targets.data.ne(onmt.Constants.PAD).sum()
                report_loss += loss
                report_num_correct += num_correct
                report_tgt_words += num_words
                total_loss += loss
                total_num_correct += num_correct
                total_words += num_words
            report_src_words += batch.lengths.data.sum()

            if i % opt.log_interval == -1 % opt.log_interval:
                ppl = math.exp(report_loss / report_tgt_words)
                acc = report_num_correct / report_tgt_words * 100
                tgtper = report_tgt_words/(time.time()-start + 1e-5)
                if opt.log_server:
                    experiment.add_scalar_value("ppl", ppl)
                    experiment.add_scalar_value("accuracy", acc)
                    experiment.add_scalar_value("tgtper", tgtper)
                    experiment.add_scalar_value("lr", optim.lr)
                print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f;" +
                       "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
                      (epoch, i+1, len(trainData),
                       acc,
                       ppl,
                       report_src_words/(time.time()-start + 1e-5),
                       tgtper,
                       time.time()-start_time))
                sys.stdout.flush()
                report_loss, report_tgt_words = 0, 0
                report_src_words, report_num_correct = 0, 0

                start = time.time()

        return total_loss / total_words, total_num_correct / total_words

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_loss, train_acc = trainEpoch(epoch)
        train_ppl = math.exp(min(train_loss, 100))
        print('Train perplexity: %g' % train_ppl)
        print('Train accuracy: %g' % (train_acc*100))

        #  (2) evaluate on the validation set
        valid_loss, valid_acc = eval(model, criterion, validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        print('Validation accuracy: %g' % (valid_acc*100))

        if opt.log_server:
            experiment.add_scalar_value("train_ppl", train_ppl)
            experiment.add_scalar_value("train_acc", train_acc*100)
            experiment.add_scalar_value("valid_ppl", valid_ppl)
            experiment.add_scalar_value("valid_acc", valid_acc*100)

        #  (3) update the learning rate
        optim.updateLearningRate(valid_ppl, epoch)

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
                       % (opt.save_model, 100*valid_acc, valid_ppl, epoch))


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

    model = onmt.Models.NMTModel(encoder, decoder)

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
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

    model.generator = generator

    if not opt.train_from_state_dict and not opt.train_from:
        if opt.param_init != 0.0:
            print('Intializing params')
            for p in model.parameters():
                p.data.uniform_(-opt.param_init, opt.param_init)

        encoder.load_pretrained_vectors(opt)
        decoder.load_pretrained_vectors(opt)

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
