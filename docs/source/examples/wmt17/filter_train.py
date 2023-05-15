#!/usr/bin/env python3

# Usage: python3 filter_train.py in.src in.trg out.src out.trg max-tokens

import sys

if len(sys.argv[1:]) >= 4:
    in_src_fname, in_trg_fname, out_src_fname, out_trg_fname = sys.argv[1:5]
else:
    in_src_fname, in_trg_fname, out_src_fname, out_trg_fname = (
        "train.src.bpe",
        "train.trg.bpe",
        "train.src.bpe.filter",
        "train.trg.bpe.filter",
    )

max_tokens = int(sys.argv[5]) if len(sys.argv[1:]) > 4 else 95

with open(in_src_fname, mode="rt", encoding="utf-8") as in_src, open(
    in_trg_fname, mode="rt", encoding="utf-8"
) as in_trg, open(out_src_fname, mode="wt", encoding="utf-8") as out_src, open(
    out_trg_fname, mode="wt", encoding="utf-8"
) as out_trg:
    for src_line, trg_line in zip(in_src, in_trg):
        if len(src_line.split()) <= max_tokens and len(trg_line.split()) <= max_tokens:
            print(src_line, file=out_src, end="")
            print(trg_line, file=out_trg, end="")
