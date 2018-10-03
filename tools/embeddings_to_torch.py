#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import six
import sys
import numpy as np
import argparse
import torch
from onmt.utils.logging import init_logger, logger


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
    assert enc_vocab is not None and dec_vocab is not None

    logger.info("From: %s" % dict_file)
    logger.info("\t* source vocab: %d words" % len(enc_vocab))
    logger.info("\t* target vocab: %d words" % len(dec_vocab))

    return enc_vocab, dec_vocab


def get_embeddings(file_enc, opt, flag):
    embs = dict()
    if flag == 'enc':
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
            embs[l_split[0]] = [float(em) for em in l_split[1:]]
        logger.info("Got {} encryption embeddings from {}".format(len(embs),
                                                                  file_enc))
    else:

        for (i, l) in enumerate(open(file_enc, 'rb')):
            if not l:
                break
            if len(l) == 0:
                continue

            l_split = l.decode('utf8').strip().split(' ')
            if len(l_split) == 2:
                continue
            embs[l_split[0]] = [float(em) for em in l_split[1:]]
        logger.info("Got {} decryption embeddings from {}".format(len(embs),
                                                                  file_enc))
    return embs


def match_embeddings(vocab, emb, opt):
    dim = len(six.next(six.itervalues(emb)))
    filtered_embeddings = np.zeros((len(vocab), dim))
    count = {"match": 0, "miss": 0}
    for w, w_id in vocab.stoi.items():
        if w in emb:
            filtered_embeddings[w_id] = emb[w]
            count['match'] += 1
        else:
            if opt.verbose:
                logger.info(u"not found:\t{}".format(w), file=sys.stderr)
            count['miss'] += 1

    return torch.Tensor(filtered_embeddings), count


TYPES = ["GloVe", "word2vec"]


def main():

    parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
    parser.add_argument('-emb_file_enc', required=True,
                        help="source Embeddings from this file")
    parser.add_argument('-emb_file_dec', required=True,
                        help="target Embeddings from this file")
    parser.add_argument('-output_file', required=True,
                        help="Output file for the prepared data")
    parser.add_argument('-dict_file', required=True,
                        help="Dictionary file")
    parser.add_argument('-verbose', action="store_true", default=False)
    parser.add_argument('-skip_lines', type=int, default=0,
                        help="Skip first lines of the embedding file")
    parser.add_argument('-type', choices=TYPES, default="GloVe")
    opt = parser.parse_args()

    enc_vocab, dec_vocab = get_vocabs(opt.dict_file)
    if opt.type == "word2vec":
        opt.skip_lines = 1

    embeddings_enc = get_embeddings(opt.emb_file_enc, opt, flag='enc')
    embeddings_dec = get_embeddings(opt.emb_file_dec, opt, flag='dec')

    filtered_enc_embeddings, enc_count = match_embeddings(enc_vocab,
                                                          embeddings_enc,
                                                          opt)
    filtered_dec_embeddings, dec_count = match_embeddings(dec_vocab,
                                                          embeddings_dec,
                                                          opt)
    logger.info("\nMatching: ")
    match_percent = [_['match'] / (_['match'] + _['miss']) * 100
                     for _ in [enc_count, dec_count]]
    logger.info("\t* enc: %d match, %d missing, (%.2f%%)"
                % (enc_count['match'],
                   enc_count['miss'],
                   match_percent[0]))
    logger.info("\t* dec: %d match, %d missing, (%.2f%%)"
                % (dec_count['match'],
                   dec_count['miss'],
                   match_percent[1]))

    logger.info("\nFiltered embeddings:")
    logger.info("\t* enc: %s" % str(filtered_enc_embeddings.size()))
    logger.info("\t* dec: %s" % str(filtered_dec_embeddings.size()))

    enc_output_file = opt.output_file + ".enc.pt"
    dec_output_file = opt.output_file + ".dec.pt"
    logger.info("\nSaving embedding as:\n\t* enc: %s\n\t* dec: %s"
                % (enc_output_file, dec_output_file))
    torch.save(filtered_enc_embeddings, enc_output_file)
    torch.save(filtered_dec_embeddings, dec_output_file)
    logger.info("\nDone.")


if __name__ == "__main__":
    init_logger('embeddings_to_torch.log')
    main()
