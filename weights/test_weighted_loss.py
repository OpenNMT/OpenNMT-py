import torch
import torch.nn as nn
from torch.autograd import Variable
from onmt.Loss import DatumWeightedNLLCriterion, NMTLossComputeDatumWeighted
import onmt
import opts
import argparse
import sys
import os
# from weighted_loss import NMTLossComputeDatumWeighted


def create_variables(batch_size, cats):
    # loss receives the batches... 32 x 64 x 500
    # crit iput calculation is [a, dict_size]
    # target is [a]?

    # create "scores"
    scores = torch.rand((batch_size, cats)).log()
    scores = torch.autograd.Variable(scores)

    # create "targets"
    targets_base = torch.arange(0, min(batch_size, cats), 1).long()
    targets = targets_base.clone()
    for _ in range(cats, batch_size, cats):
        missing = batch_size - targets.size()[0]
        targets = torch.cat((targets, targets_base[:missing]), 0)
    targets = torch.autograd.Variable(targets)

    # create "datum weights"
    dw_1 = torch.ones(batch_size).float()
    # dw = torch.normal(torch.zeros(batch_size), torch.ones(batch_size))
    dw = torch.rand(batch_size)
    dw = dw.float()

    return scores, targets, dw, dw_1


# def eq_crit(scores, targets):
#     crit = DatumWeightedNLLloss(weight, size_average=False)

def test_criterions():
    batch_size = 10
    cats= 5
    scores, targets, dw, dw_1 = create_variables(batch_size, cats)

    # standar crit
    weight = torch.arange(0, cats, 1).float()
    dim = scores.dim()
    crit = nn.NLLLoss(weight, size_average=False)

    # out [32, 64, 500]

    # loss = nn.NLLLoss()
    # crit(scores, target) | (scores, align, target) if there is copy_attention
    # scores.size([2048, 50000]) -> out from the model, log-probabilities for each class
    # target.size([2048]) -> class N (base 0)
    # _assert_no_grad(target)

    result = crit(scores, targets)
    eq_crit_w1 = DatumWeightedNLLCriterion(weight, size_average=False)
    eq_crit_w = DatumWeightedNLLCriterion(weight, size_average=False, datum_average=False)
    result_eq_w1 = eq_crit_w1(scores, targets, dw_1)
    result_eq_w = eq_crit_w(scores, targets, dw)
    print(result)
    print(result_eq_w1)
    print(result_eq_w)
    # TypeError: FloatClassNLLCriterion_updateOutput received an invalid combination of arguments -
    # got (int, torch.FloatTensor, !torch.FloatTensor!, torch.FloatTensor, bool, NoneType, torch.FloatTensor, int),
    # but expected
    # (int state, torch.FloatTensor input, torch.LongTensor target, torch.FloatTensor output, bool sizeAverage,
    #   [torch.FloatTensor weights or None], torch.FloatTensor total_weight, int ignore_index)


def test_loss_object(opt):
    # Load train and validate data.
    # opt = None
    print("Loading train and validate data from '%s'" % opt.data)
    train = torch.load(opt.data + '.train.pt')
    valid = torch.load(opt.data + '.valid.pt')
    checkpoint = None
    model_opt = opt
    fields = load_fields(train, valid, checkpoint)
    model = build_model(model_opt, opt, fields, checkpoint)

    # make loss compute object
    compute_orig = onmt.Loss.NMTLossCompute(model.generator, fields["tgt"].vocab)
    compute_w1 = NMTLossComputeDatumWeighted(model.generator, fields["tgt"].vocab)
    compute_w = NMTLossComputeDatumWeighted(model.generator, fields["tgt"].vocab)

    # generate dummy model outputs
    scores, targets, dw, dw_1 = create_variables(1000, len(fields["tgt"].vocab))

    # compute losses
    cost = compute_orig.criterion(scores, targets) # **kwarg
    cost_w1 = compute_w1.criterion(scores, targets, dw_1)
    cost_w = compute_w.criterion(scores, targets, dw)

    print(cost, cost_w1, cost_w)


def preprocess_mock(root_dir):
    print(" ".join(["python", os.path.join(root_dir, "preprocess.py"),
         "-train_src", os.path.join(root_dir, "data", "src-train.txt"),
         "-train_tgt", os.path.join(root_dir, "data", "tgt-train.txt"),
         "-valid_src", os.path.join(root_dir, "data", "src-val.txt"),
         "-valid_tgt", os.path.join(root_dir, "data", "tgt-val.txt"),
         "-train_dw", os.path.join(root_dir, "data", "dw-train.txt"),
         "-valid_dw", os.path.join(root_dir, "data", "dw-val.txt"),
         "-save_data", os.path.join(root_dir, "weights", "tester.p")]))

if __name__ == "__main__":
    print(os.path.abspath(os.path.curdir))
    root_dir = os.path.dirname(os.path.abspath(os.path.curdir))

    # the string to execute in the correct environment
    # preprocess_mock(root_dir)

    sys.argv.extend(["-data", "../weights/tester.p"])
    #
    from train import load_fields, build_model
    parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(parser)
    opts.train_opts(parser)
    opt = parser.parse_args()
    print(opt)
    # test_criterions()
    test_loss_object(opt)
