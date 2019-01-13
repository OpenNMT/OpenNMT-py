from __future__ import division
import torch
from collections import Counter


def bleu_calculations(predictions, gold_targets, ngram,
                      exclude_indices=None):
    _precision_matches = Counter()
    _precision_totals = Counter()
    _prediction_lengths = 0
    _reference_lengths = 0

    predictions = torch.transpose(predictions, 0, 1)
    gold_targets = torch.transpose(gold_targets, 0, 1)

    predictions, gold_targets = unwrap_to_tensors(
        predictions, gold_targets)
    for ngram_size in range(1, ngram + 1):
        precision_matches, precision_totals = \
            _get_modified_precision_counts(
                predictions, gold_targets, ngram_size, exclude_indices)
        _precision_matches[ngram_size] += precision_matches
        _precision_totals[ngram_size] += precision_totals

    if exclude_indices is None:
        _prediction_lengths += predictions.size(0) * \
            predictions.size(1)
        _reference_lengths += gold_targets.size(0) * \
            gold_targets.size(1)
    else:
        valid_predictions_mask = _get_valid_tokens_mask(
            predictions, exclude_indices)
        _prediction_lengths += valid_predictions_mask.sum().item()
        valid_gold_targets_mask = _get_valid_tokens_mask(
            gold_targets, exclude_indices)
        _reference_lengths += valid_gold_targets_mask.sum().item()
    return _precision_matches, _precision_totals, _prediction_lengths, \
        _reference_lengths


def unwrap_to_tensors(*tensors):
    """
    If you actually passed gradient-tracking Tensors to this,
    there will be a huge memory leak, because it will prevent
    garbage collection for the computation graph. This method
    ensures that you're using tensors directly and that they
    are on the CPU.
    """
    return (x.detach().cpu() if isinstance(x, torch.Tensor)
            else x for x in tensors)


def _get_modified_precision_counts(
                                   predicted_tokens,
                                   reference_tokens,
                                   ngram_size,
                                   exclude_indices=None):
    """
    Compare the predicted tokens to the reference (gold) tokens
    at the desired ngram size and calculate the numerator and
    denominator for a modified form of precision.

    The numerator is the number of ngrams in the predicted
    sentences that match with an ngram in the corresponding
    reference sentence, clipped by the total count of that
    ngram in the reference sentence. The denominator is just
    the total count of predicted ngrams.
    """
    clipped_matches = 0
    total_predicted = 0
    for batch_num in range(predicted_tokens.size(0)):
        predicted_row = predicted_tokens[batch_num, :]
        reference_row = reference_tokens[batch_num, :]
        predicted_ngram_counts = _ngrams(
            predicted_row, ngram_size, exclude_indices)
        reference_ngram_counts = _ngrams(
            reference_row, ngram_size, exclude_indices)
        for ngram, count in predicted_ngram_counts.items():
            clipped_matches += min(
                count, reference_ngram_counts[ngram])
            total_predicted += count
    return clipped_matches, total_predicted


def _ngrams(
            tensor,
            ngram_size,
            exclude_indices=None):
    ngram_counts = Counter()
    if ngram_size > tensor.size(-1):
        return ngram_counts
    for start_position in range(ngram_size):
        for tensor_slice in tensor[start_position:].split(
                ngram_size, dim=-1):
            if tensor_slice.size(-1) < ngram_size:
                break
            ngram = tuple(x.item() for x in tensor_slice)
            if any(x in exclude_indices for x in ngram):
                continue
            ngram_counts[ngram] += 1
    return ngram_counts


def _get_valid_tokens_mask(tensor,
                           exclude_indices=None):
    valid_tokens_mask = torch.ones(tensor.size(), dtype=torch.uint8)
    for index in exclude_indices:
        valid_tokens_mask = valid_tokens_mask & (tensor != index)
    return valid_tokens_mask
