#!/usr/bin/env python3
# coding: utf-8

import sacrebleu
import codecs
from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("--nbest-hyp", type=str, help="file with nbest to rerank",
                    required=True)
parser.add_argument("--nbest-ref", type=str, help="ref repeated n times",
                    required=True)
parser.add_argument("--nbest-order", type=int, help="nbest order",
                    required=True)
parser.add_argument("--output", type=str, help="output file", required=True)

args = parser.parse_args()


def chunks(lgth, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(lgth), n):
        yield lgth[i:i + n]


with codecs.open(args.nbest_hyp, encoding="utf-8") as file:
    nbests = file.readlines()
    nbests = [item.strip() for item in nbests]
    nbests = chunks(nbests, args.nbest_order)

with codecs.open(args.nbest_ref, encoding="utf-8") as file:
    nrefs = file.readlines()
    nrefs = [item.strip() for item in nrefs]
    nrefs = chunks(nrefs, args.nbest_order)

with codecs.open(args.output, "w", encoding="utf-8") as output_file:
    best_indices = []
    for nbest, nref in zip(nbests, nrefs):
        texts = []
        scores = []
        for hyp, gold in zip(nbest, nref):
            bleu = sacrebleu.sentence_bleu(hyp, [gold]).score
            texts.append(hyp)
            scores.append(bleu)
            max_index = scores.index(max(scores))
        output_file.write(texts[max_index] + "\n")
        best_indices.append(max_index)

for i in range(args.nbest_order):
    print(i, best_indices.count(i))
