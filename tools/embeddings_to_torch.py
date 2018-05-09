#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import six
import numpy as np
import argparse
import torch
import itertools
from collections import defaultdict
import logging as log
log.basicConfig(level=log.INFO)


def get_vocabs(dict_file):
    vocabs = torch.load(dict_file)

    enc_vocab, dec_vocab = None, None

    # the vocab object is a list of tuple (name, torchtext.Vocab)
    # we iterate over this list and associate vocabularies based on the name
    for vocab in vocabs:
        if vocab[0] == 'src':
            enc_vocab = vocab[1]
        if vocab[0] == 'tgt':
            dec_vocab = vocab[1]
    assert type(None) not in [type(enc_vocab), type(dec_vocab)]

    print("From: %s" % dict_file)
    print("\t* source vocab: %d words" % len(enc_vocab))
    print("\t* target vocab: %d words" % len(dec_vocab))

    return enc_vocab, dec_vocab


def get_embeddings(file_enc, opt):
    log.info("Reading %s" % file_enc)
    count = 0
    for (i, l) in enumerate(open(file_enc, 'rb')):
        if i < opt.skip_lines:
            continue
        if not l:
            break
        if len(l) == 0:
            continue
        l_split = l.decode('utf8').strip().split(' ')
        if len(l_split) == 2:
            continue
        count += 1
        word, vec = l_split[0], list(map(float, l_split[1:]))
        yield word, vec

    log.info("Got {} encryption embeddings from {}".format(count, file_enc))


def match_embeddings(vocab, emb, opt, side):

    # peeking first item to know its dimension
    first_word, first_vec = six.next(emb)
    dim = len(first_vec)
    log.info("%s dimensions %d" % (side, dim))
    # put the peeked record back it to generator stream
    emb = itertools.chain([(first_word, first_vec)], emb)

    filtered_embeddings = np.zeros((len(vocab), dim))
    count = defaultdict(int)
    matched_types = set()
    ignored_types = set()
    for w, w_vec in emb:
        count['tot_emb'] += 1
        if w in vocab.stoi:
            w_id = vocab.stoi[w]
            filtered_embeddings[w_id] = w_vec
            count['match'] += 1
            matched_types.add(w)
        else:
            ignored_types.add(w)
    count['miss'] = len(vocab) - count['match']

    log.info("Matching: ")
    match_percent = count['match'] / (len(vocab)) * 100
    log.info("* %s: %d match, %d missing, (%.2f%%)" % (
        side, count['match'], count['miss'], match_percent))

    if opt.verbose:
        for name, types in [('missed', set(vocab.stoi.keys()) - matched_types),
                            ('matched', matched_types),
                            ('ignored', ignored_types)]:
            path = opt.output_file + '.%s.%s.txt' % (side, name)
            log.info("Writing %d %s types to %s" % (len(types), name, path))
            write_lines(types, path)
    return torch.Tensor(filtered_embeddings), count


def write_lines(items, path):
    with open(path, 'w', encoding='utf-8') as fw:
        for line in items:
            fw.write(line)
            fw.write('\n')


TYPES = ["GloVe", "word2vec"]


def main():

    parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
    parser.add_argument('-emb_file_enc',
                        help="source Embeddings from this file")
    parser.add_argument('-emb_file_dec',
                        help="target Embeddings from this file")
    parser.add_argument('-output_file', required=True,
                        help="Output file for the prepared data")
    parser.add_argument('-dict_file', required=True,
                        help="Dictionary file")
    parser.add_argument('-verbose', action="store_true",
                        help='produce matched, ignored, missed types report')
    parser.add_argument('-skip_lines', type=int, default=0,
                        help="Skip first lines of the embedding file")
    parser.add_argument('-type', choices=TYPES, default="GloVe")
    opt = parser.parse_args()
    if not opt.emb_file_enc and not opt.emb_file_dec:
        raise Exception('Either or both required: emb_file_enc, -emb_file_dec')

    enc_vocab, dec_vocab = get_vocabs(opt.dict_file)
    if opt.type == "word2vec":
        opt.skip_lines = 1

    if opt.emb_file_enc:
        embeddings_enc = get_embeddings(opt.emb_file_enc, opt)
        filtered_enc_embeddings, enc_count = match_embeddings(
            enc_vocab, embeddings_enc, opt, 'enc')

        enc_output_file = opt.output_file + ".enc.pt"
        log.info("Saving enc embeddings {} as: {}".format(
            filtered_enc_embeddings.size(), enc_output_file))
        torch.save(filtered_enc_embeddings, enc_output_file)
        del filtered_enc_embeddings     # free memory

    if opt.emb_file_dec:
        embeddings_dec = get_embeddings(opt.emb_file_dec, opt)
        filtered_dec_embeddings, dec_count = match_embeddings(
            dec_vocab, embeddings_dec, opt, 'dec')
        dec_output_file = opt.output_file + ".dec.pt"
        log.info("Saving dec embeddings {} as: {}" .format(
            filtered_dec_embeddings.size(), dec_output_file))
        torch.save(filtered_dec_embeddings, dec_output_file)
    print("\nDone.")


if __name__ == "__main__":
    main()
