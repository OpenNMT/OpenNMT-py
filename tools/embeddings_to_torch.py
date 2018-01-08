#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import six
import sys
import numpy as np
import argparse
import torch

parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
parser.add_argument('-emb_file', required=True,
                    help="Embeddings from this file")
parser.add_argument('-output_file', required=True,
                    help="Output file for the prepared data")
parser.add_argument('-dict_file', required=True,
                    help="Dictionary file")
parser.add_argument('-verbose', action="store_true", default=False)
opt = parser.parse_args()


def get_vocabs(dict_file):
    vocabs = torch.load(dict_file)
    enc_vocab, dec_vocab = vocabs[0][1], vocabs[-1][1]

    print("From: %s" % dict_file)
    print("\t* source vocab: %d words" % len(enc_vocab))
    print("\t* target vocab: %d words" % len(dec_vocab))

    return enc_vocab, dec_vocab


def get_embeddings(file):
    embs = dict()
    for l in open(file, 'rb').readlines():
        l_split = l.decode('utf8').strip().split()
        if len(l_split) == 2:
            continue
        embs[l_split[0]] = [float(em) for em in l_split[1:]]
    print("Got {} embeddings from {}".format(len(embs), file))

    return embs


def match_embeddings(vocab, emb):
    dim = len(six.next(six.itervalues(emb)))
    filtered_embeddings = np.zeros((len(vocab), dim))
    count = {"match": 0, "miss": 0}
    for w, w_id in vocab.stoi.items():
        if w in emb:
            filtered_embeddings[w_id] = emb[w]
            count['match'] += 1
        else:
            if opt.verbose:
                print(u"not found:\t{}".format(w), file=sys.stderr)
            count['miss'] += 1

    return torch.Tensor(filtered_embeddings), count


def main():
    enc_vocab, dec_vocab = get_vocabs(opt.dict_file)
    embeddings = get_embeddings(opt.emb_file)

    filtered_enc_embeddings, enc_count = match_embeddings(enc_vocab,
                                                          embeddings)
    filtered_dec_embeddings, dec_count = match_embeddings(dec_vocab,
                                                          embeddings)

    print("\nMatching: ")
    match_percent = [_['match'] / (_['match'] + _['miss']) * 100
                     for _ in [enc_count, dec_count]]
    print("\t* enc: %d match, %d missing, (%.2f%%)" % (enc_count['match'],
                                                       enc_count['miss'],
                                                       match_percent[0]))
    print("\t* dec: %d match, %d missing, (%.2f%%)" % (dec_count['match'],
                                                       dec_count['miss'],
                                                       match_percent[1]))

    print("\nFiltered embeddings:")
    print("\t* enc: ", filtered_enc_embeddings.size())
    print("\t* dec: ", filtered_dec_embeddings.size())

    enc_output_file = opt.output_file + ".enc.pt"
    dec_output_file = opt.output_file + ".dec.pt"
    print("\nSaving embedding as:\n\t* enc: %s\n\t* dec: %s"
          % (enc_output_file, dec_output_file))
    torch.save(filtered_enc_embeddings, enc_output_file)
    torch.save(filtered_dec_embeddings, dec_output_file)
    print("\nDone.")


if __name__ == "__main__":
    main()
