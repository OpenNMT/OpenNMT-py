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
from train_opts import add_model_arguments

parser = argparse.ArgumentParser(description='translate.py')
onmt.Markdown.add_md_help_argument(parser)

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
# Most relevant to copy model that can generate OOV tokens
parser.add_argument('-decorate_oov', action='store_true',
                    help='Decorate OOV tokens in verbose printout.')

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


def decorate_oov(tokens, vocab):
    for i, token in enumerate(tokens):
        if vocab.stoi[token] == 0:
            # token is OOV
            tokens[i] = '__%s__' % token
    return tokens


def main():
    opt = parser.parse_args()
    dummy_parser = argparse.ArgumentParser(description='train.py')
    add_model_arguments(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args()[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    translator = onmt.Translator(opt, dummy_opt.__dict__)
    outF = codecs.open(opt.output, 'w', 'utf-8')
    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0
    srcBatch, tgtBatch = [], []
    count = 0
    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()

    data = onmt.IO.ONMTDataset(opt.src, opt.tgt, translator.fields, None)

    testData = onmt.IO.OrderedIterator(
        dataset=data, device=opt.gpu if opt.gpu else -1,
        batch_size=opt.batch_size, train=False, sort=False,
        shuffle=False)

    # index = 0
    for batch in testData:
        predBatch, predScore, goldScore, attn, src \
            = translator.translate(batch, data)
        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)
        if opt.tgt:
            goldScoreTotal += sum(goldScore)
            goldWordsTotal += sum(len(x) for x in tgtBatch)

        for b in range(len(predBatch)):
            count += 1
            try:
                # python2
                outF.write(" ".join([i.decode('utf-8')
                           for i in predBatch[b][0]]) + '\n')
            except AttributeError:
                # python3: can't do .decode on a str object
                outF.write(" ".join(predBatch[b][0]) + '\n')
            outF.flush()

            if opt.verbose:
                # print(srcBatch[b])
                # srcSent = ' '.join(srcBatch[b])
                # if translator.tgt_dict.lower:
                #     srcSent = srcSent.lower()

                if opt.decorate_oov:
                    example_index = batch.indices.data[b]
                    words = decorate_oov(data[example_index].src,
                                         translator.fields["src"].vocab)
                else:
                    words = []
                    for f in src[:, b]:
                        word = translator.fields["src"].vocab.itos[f]
                        if word == onmt.IO.PAD_WORD:
                            break
                        words.append(word)

                os.write(1, bytes('SENT %d: %s\n' %
                                  (count, " ".join(words)), 'UTF-8'))
                # ex = data.examples[index]
                # print(index, list(zip(ex.src, ex.src_feat_0, ex.src_feat_1,
                #                       ex.src_feat_2)))

                # index += 1

                if opt.decorate_oov:
                    tokens = decorate_oov(predBatch[b][0],
                                          translator.fields['tgt'].vocab)
                else:
                    tokens = predBatch[b][0]

                os.write(1, bytes('\nPRED %d (len = %d): %s\n' %
                                  (count, len(tokens),
                                   " ".join(tokens)), 'UTF-8'))
                print("PRED SCORE: %.4f\n" % predScore[b][0])

                if opt.tgt:
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
                                 " ".join(predBatch[b][n])),
                            'UTF-8'))

                if opt.attn_debug:
                    print('')
                    for i, w in enumerate(predBatch[b][0]):
                        print(w)
                        _, ids = attn[b][0][i].sort(0, descending=True)
                        for j in ids[:5].tolist():
                            print("\t%s\t%d\t%3f" % (srcBatch[b][j], j,
                                                     attn[b][0][i][j]))

        srcBatch, tgtBatch = [], []

    reportScore('PRED', predScoreTotal, predWordsTotal)
    if opt.tgt:
        reportScore('GOLD', goldScoreTotal, goldWordsTotal)

    if opt.dump_beam:
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))


if __name__ == "__main__":
    main()
