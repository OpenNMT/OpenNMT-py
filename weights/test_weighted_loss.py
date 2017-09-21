import torch
import torch.nn as nn
from torch.autograd import Variable
from weights.weighted_loss import DatumWeightedNLLloss


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
    dw = torch.normal(torch.zeros(batch_size), torch.ones(batch_size))
    dw = dw.float()

    return scores, targets, dw, dw_1


# def eq_crit(scores, targets):
#     crit = DatumWeightedNLLloss(weight, size_average=False)

if __name__ == "__main__":
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
    eq_crit_w1 = DatumWeightedNLLloss(weight, size_average=False)
    eq_crit_w = DatumWeightedNLLloss(weight, size_average=False, datum_average=False)
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
