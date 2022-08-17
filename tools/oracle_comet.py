#!/usr/bin/env python3
# coding: utf-8

import codecs
from argparse import ArgumentParser
from comet import download_model, load_from_checkpoint

model_path = download_model("wmt21-comet-qe-mqm")
model = load_from_checkpoint(model_path)

parser = ArgumentParser()

parser.add_argument("--nbest-src", type=str, help="src repeated n times",
                    required=False)
parser.add_argument("--nbest-hyp", type=str, help="file with nbest to rerank",
                    required=True)
# parser.add_argument("--nbest-ref", type=str, help="ref repeated n times",
#                     required=False)
parser.add_argument("--nbest-order", type=int, help="nbest order",
                    required=True)
parser.add_argument("--output", type=str, help="output file", required=True)

args = parser.parse_args()


def chunks(lgth, n):
    """Yield successive n-sized chunks from lgth."""
    for i in range(0, len(lgth), n):
        yield lgth[i:i + n]


with codecs.open(args.nbest_hyp, encoding="utf-8") as file:
    nbests = file.readlines()
    nbests = [item.strip() for item in nbests]
    nbests = chunks(nbests, args.nbest_order)

# with codecs.open(args.nbest_ref, encoding="utf-8") as file:
#    nrefs = file.readlines()
#    nrefs = [item.strip() for item in nrefs]
#    nrefs = chunks(nrefs, args.nbest_order)

with codecs.open(args.nbest_src, encoding="utf-8") as file:
    nsrcs = file.readlines()
    nsrcs = [item.strip() for item in nsrcs]
    nsrcs = chunks(nsrcs, args.nbest_order)


with codecs.open(args.output, "w", encoding="utf-8") as output_file:
    best_indices = []
    # for nbest in nbests:
    # for nsrc, nbest, nref in zip(nsrcs, nbests, nrefs):
    for nsrc, nbest in zip(nsrcs, nbests):
        texts = []
        scores = []
        data = []
        for i in range(args.nbest_order):
            data.append({
                "src": nsrc[i],
                "mt": nbest[i],
                # "ref": nref[i],
                })
            seg_scores, sys_score = model.predict(data,
                                                  batch_size=args.nbest_order,
                                                  gpus=1)
            max_index = seg_scores.index(max(seg_scores))
            output_file.write(data[max_index]["mt"] + "\n")
            best_indices.append(max_index)

for i in range(args.nbest_order):
    print(i, best_indices.count(i))
