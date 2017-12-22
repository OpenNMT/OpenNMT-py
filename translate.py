#!/usr/bin/env python

from __future__ import division, unicode_literals
import os
import argparse
import math
import codecs
import torch

import onmt
import onmt.io
import onmt.translate
import opts
from itertools import takewhile, count

from six.moves import zip_longest
from six.moves import zip

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()


def report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total/words_total)))


def get_src_words(src_indices, index2str):
    raw_words = (index2str[i] for i in src_indices)
    words = takewhile(lambda w: w != onmt.io.PAD_WORD, raw_words)
    return " ".join(words)


def main():

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    translator = onmt.translate.Translator(opt, dummy_opt.__dict__)
    out_file = codecs.open(opt.output, 'w', 'utf-8')
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()

    data = onmt.io.build_dataset(translator.fields, opt.data_type,
                                 opt.src, opt.tgt,
                                 src_dir=opt.src_dir,
                                 sample_rate=opt.sample_rate,
                                 window_size=opt.window_size,
                                 window_stride=opt.window_stride,
                                 window=opt.window,
                                 use_filter_pred=False)
    data_type = data.data_type

    test_data = onmt.io.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        shuffle=False)

    counter = count(1)
    for batch in test_data:
        pred_batch, gold_batch, pred_scores, gold_scores, attn, src, indices\
            = translator.translate(batch, data)
        pred_score_total += sum(score[0] for score in pred_scores)
        pred_words_total += sum(len(x[0]) for x in pred_batch)
        if opt.tgt:
            gold_score_total += sum(gold_scores)
            gold_words_total += sum(len(x) for x in batch.tgt[1:])

        # z_batch: an iterator over the predictions, their scores,
        # the gold sentence, its score, and the source sentence for each
        # sentence in the batch. It has to be zip_longest instead of
        # plain-old zip because the gold_batch has length 0 if the target
        # is not included.
        if data_type == 'text':
            sents = src.split(1, dim=1)
        else:
            sents = [torch.Tensor(1, 1) for i in range(len(pred_scores))]
        z_batch = zip_longest(
                pred_batch, gold_batch,
                pred_scores, gold_scores,
                (sent.squeeze(1) for sent in sents), indices)

        for pred_sents, gold_sent, pred_score, gold_score, src_sent, index\
                in z_batch:
            n_best_preds = [" ".join(pred) for pred in pred_sents[:opt.n_best]]
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()

            if opt.verbose:
                sent_number = next(counter)
                if data_type == 'text':
                    words = get_src_words(
                        src_sent, translator.fields["src"].vocab.itos)
                else:
                    words = test_data.dataset.examples[index].src_path

                output = '\nSENT {}: {}\n'.format(sent_number, words)
                os.write(1, output.encode('utf-8'))

                best_pred = n_best_preds[0]
                best_score = pred_score[0]
                output = 'PRED {}: {}\n'.format(sent_number, best_pred)
                os.write(1, output.encode('utf-8'))
                print("PRED SCORE: {:.4f}".format(best_score))

                if opt.tgt:
                    tgt_sent = ' '.join(gold_sent)
                    output = 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
                    os.write(1, output.encode('utf-8'))
                    print("GOLD SCORE: {:.4f}".format(gold_score))

                if len(n_best_preds) > 1:
                    print('\nBEST HYP:')
                    for score, sent in zip(pred_score, n_best_preds):
                        output = "[{:.4f}] {}\n".format(score, sent)
                        os.write(1, output.encode('utf-8'))

    report_score('PRED', pred_score_total, pred_words_total)
    if opt.tgt:
        report_score('GOLD', gold_score_total, gold_words_total)

    if opt.dump_beam:
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))


if __name__ == "__main__":
    main()
