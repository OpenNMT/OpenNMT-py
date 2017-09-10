from __future__ import division
from builtins import bytes
import os
import argparse
import math
import codecs
import torch

import onmt
import onmt.IO
import opts

parser = argparse.ArgumentParser(description='translate.py')
opts.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-src_img_dir',   default="",
                    help='Source image directory')
parser.add_argument('-tgt',
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
# options most relevant to summarization
parser.add_argument('-dynamic_dict', action='store_true',
                    help="Create dynamic dictionaries")
parser.add_argument('-share_vocab', action='store_true',
                    help="Share source and target vocabulary")


def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))


def main():
    opt = parser.parse_args()

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    translator = onmt.Translator(opt, dummy_opt.__dict__)
    outF = codecs.open(opt.output, 'w', 'utf-8')
    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0
    count = 0
    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()
    data = onmt.IO.ONMTDataset(opt.src, opt.tgt, translator.fields, None)

    testData = onmt.IO.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        shuffle=False)

    index = 0
    for batch in testData:
        predBatch, goldBatch, predScore, goldScore, attn, src \
            = translator.translate(batch, data)
        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)
        if opt.tgt:
            goldScoreTotal += sum(goldScore)
            goldWordsTotal += sum(len(x) for x in batch.tgt[1:])

        for b in range(len(predBatch)):
            count += 1
            try:
                # python2 (should be the same)
                for n in range(opt.n_best):
                    outF.write(" ".join([i
                               for i in predBatch[b][n]]) + '\n')
            except AttributeError:
                # python3: can't do .decode on a str object
                for n in range(opt.n_best):
                    outF.write(" ".join(predBatch[b][n]) + '\n')
            outF.flush()

            if opt.verbose:
                words = []
                for f in src[:, b]:
                    word = translator.fields["src"].vocab.itos[f]
                    if word == onmt.IO.PAD_WORD:
                        break
                    words.append(word)

                os.write(1, bytes('\nSENT %d: %s\n' %
                                  (count, " ".join(words)), 'UTF-8'))

                index += 1
                os.write(1, bytes('PRED %d: %s\n' %
                                  (count, " ".join(predBatch[b][0])), 'UTF-8'))
                print("PRED SCORE: %.4f" % predScore[b][0])

                if opt.tgt:
                    tgtSent = ' '.join(goldBatch[b])
                    os.write(1, bytes('GOLD %d: %s\n' %
                             (count, tgtSent), 'UTF-8'))
                    print("GOLD SCORE: %.4f" % goldScore[b])

                if opt.n_best > 1:
                    print('\nBEST HYP:')
                    for n in range(opt.n_best):
                        os.write(1, bytes("[%.4f] %s\n" % (predScore[b][n],
                                 " ".join(predBatch[b][n])),
                            'UTF-8'))

    reportScore('PRED', predScoreTotal, predWordsTotal)
    if opt.tgt:
        reportScore('GOLD', goldScoreTotal, goldWordsTotal)

    if opt.dump_beam:
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))


if __name__ == "__main__":
    main()
