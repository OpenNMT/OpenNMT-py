#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse
import hashlib
import time

from onmt.translate.translator import build_translator
import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts

hash = hashlib.sha1()

def translate(opt):
    translator = build_translator(opt, report_score=True)
    translator.translate(src_path=opt.src,
                         tgt_path=opt.tgt,
                         src_dir=opt.src_dir,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug)

def run(amr, model):
    if model == 'seq':
        model_path = ''
    elif model == 'tree':
        model_path = ''
    else:
        model_path = 'models/graph_gcn_seq_cpu/amr-model_step_10000.pt'

    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    hash.update(str(time.time()).encode('utf-8'))
    hash.update(amr.encode('utf-8'))
    tmp_file_path = 'amrtotext_' + hash.hexdigest() + '.txt'
    tmp_file = open(tmp_file_path, 'w')
    tmp_file.write(amr + '\n')
    tmp_file.close()

    opt = parser.parse_args()
    opt.models = [model_path]
    opt.data_type = 'amr'
    opt.reentrancies = True if model == 'graph' else False
    opt.src = tmp_file_path
    opt.output = tmp_file_path + '.pred'
    opt.beam_size = 5
    opt.batch_size = 1
    opt.max_length = 125
    print(opt.models) 
    translate(opt)

print(run('(e / eat-01)', 'graph'))
