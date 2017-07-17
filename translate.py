from __future__ import division
from builtins import bytes

import onmt
import onmt.Markdown
import onmt.IO
import torch
import argparse
import math
import codecs
import os
from itertools import zip_longest, repeat

parser = argparse.ArgumentParser(description='translate.py')
onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-src_img_dir',   default="",
                    help='Source image directory')
parser.add_argument('-tgt',
                    default=None,
                    help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
# parser.add_argument('-phrase_table',
#                     help="""Path to source-target dictionary to replace UNK
#                     tokens. See README.md for the format of this file.""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-attn_debug', action="store_true",
                    help='Print best attn for each word')

parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')

parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")

parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))

def batchify(lines, batch_size):
    """
    cf. https://docs.python.org/3/library/itertools.html#itertools-recipes
    lines is an iterable, such as the src.test lines
    """
    args = [iter(lines)] * batch_size
    for raw_batch in zip_longest(*args):
        yield tuple(line for line in raw_batch if line is not None)
        
def target_file_or_none(target):
    """
    so as to not have to check all the time if the target file exists
    """
    if target:
        return codecs.open(target, 'r', 'utf-8')
    return repeat(None)

def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = onmt.Translator(opt)

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    count = 0

    tgtF = codecs.open(opt.tgt, 'r', 'utf-8') if opt.tgt else None

    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()
        
    with codecs.open(opt.src, 'r', 'utf-8') as src_test, \
        codecs.open(opt.output, 'w', 'utf-8') as predicted_out:
        count = 1
        for src_batch in batchify(src_test, opt.batch_size):
            src_batch = [line.split() for line in src_batch]
            predicted_batch, predicted_scores, gold_scores, attn, src \
            = translator.translate(src_batch, None)
            # sort of weird that predicted batch is a list of lists of lists
            # also sort of weird that predScore is a tuple of singleton Tensors
            predScoreTotal += sum(score[0] for score in predicted_scores)
            predWordsTotal += sum(len(x[0]) for x in predicted_batch)
            # targets: no need right now: simplicity!
            '''
            if tgtF is not None:
                goldScoreTotal += sum(gold_scores)
                goldWordsTotal += sum(len(x) for x in tgtBatch)
            '''
            # 1) print the predicted lines to the outfile
            # 2) maybe do some extra things if verbose
            for src_sent, pred_sent, pred_score, gold_score in zip(src_batch, predicted_batch, predicted_scores, gold_scores):
                pred_sent = pred_sent[0]
                pred_line = " ".join(pred_sent) # breaking python2 compatibility, maybe
                predicted_out.write(pred_line + '\n')
                
                if opt.verbose:
                    src_line = " ".join(src_sent)
                    if translator.tgt_dict.lower:
                        src_line = src_line.lower()
                    os.write(1, bytes('SENT %d: %s\n' % (count, src_line), 'UTF-8'))
                    os.write(1, bytes('PRED %d: %s\n' %
                                      (count, " ".join(pred_sent)), 'UTF-8'))
                    print("PRED SCORE: %.4f" % pred_score[0])
                    """
                    if tgtF is not None:
                        tgtSent = ' '.join(tgtBatch[b])
                        if translator.tgt_dict.lower:
                            tgtSent = tgtSent.lower()
                        os.write(1, bytes('GOLD %d: %s\n' %
                                 (count, tgtSent), 'UTF-8'))
                        print("GOLD SCORE: %.4f" % goldScore[b])

                    if opt.n_best > 1:
                        print('\nBEST HYP:')
                        for n in range(opt.n_best):
                            os.write(1, bytes("[%.4f] %s\n" % (predScore[b][n],
                                     " ".join(pred_batch[n])),
                                'UTF-8'))

                    if opt.attn_debug:
                        print('')
                        for i, w in enumerate(batch[0]):
                            print(w)
                            _, ids = attn[b][0][i].sort(0, descending=True)
                            for j in ids[:5].tolist():
                                print("\t%s\t%d\t%3f" % (srcTokens[j], j,
                                                         attn[b][0][i][j]))
                    """
                count += 1
            

    reportScore('PRED', predScoreTotal, predWordsTotal)
    if tgtF:
        reportScore('GOLD', goldScoreTotal, goldWordsTotal)

    if tgtF:
        tgtF.close()

    if opt.dump_beam:
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))


if __name__ == "__main__":
    main()
