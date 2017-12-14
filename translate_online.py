#!/usr/bin/env python

from __future__ import division
import json
import argparse
import torch

import onmt
import onmt.IO
import opts

from six.moves import zip_longest
from six.moves import zip

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()


class OnlineTranslator:
    def __init__(self, translator):
        assert isinstance(translator, onmt.Translator)
        self.translator = translator

    def translate(self, sentences, device=-1,
                  batch_size=32, n_best=3):
        data = onmt.IO.ONMTDataset(sentences, None,
                                   self.translator.fields,
                                   use_filter_pred=False)

        test_data = onmt.IO.OrderedIterator(dataset=data, device=device,
                                            batch_size=batch_size, train=False,
                                            sort=False, shuffle=False)

        pred_score_total, pred_words_total = 0, 0
        sents_preds = []
        for batch in test_data:
            pred_batch, gold_batch, pred_scores, gold_scores, attn, src \
                = self.translator.translate(batch, data)
            pred_score_total += sum(score[0] for score in pred_scores)
            pred_words_total += sum(len(x[0]) for x in pred_batch)

            # z_batch: an iterator over the predictions, their scores,
            # the gold sentence, its score, and the source sentence for each
            # sentence in the batch. It has to be zip_longest instead of
            # plain-old zip because the gold_batch has length 0 if the target
            # is not included.
            z_batch = zip_longest(
                pred_batch, gold_batch,
                pred_scores, gold_scores,
                (sent.squeeze(1) for sent in src.split(1, dim=1)))

            for (pred_sents, gold_sent,
                 pred_score, gold_score, src_sent) in z_batch:
                n_best_preds = [" ".join(pred) for pred in pred_sents[:n_best]]
                sent_preds = []
                for pred, score in zip(n_best_preds, pred_score):
                    sent_preds.append({'prediction': pred,
                                       'score': score})
                sents_preds.append({'sentence': src_sent,
                                    'predictions': sents_preds})
        return sents_preds


def get_translator():
    # NOTE: I don't like that we have to do this "dummy_parser"
    # but this is how OpenNMT's Translator objects are instantiated
    # (Anton)
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    translator = onmt.Translator(opt, dummy_opt.__dict__)
    online_translator = OnlineTranslator(translator)
    return online_translator




