#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse
import hashlib
import time
import os
import subprocess

from onmt.translate.translator import build_translator
import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts

hash = hashlib.sha1()

def anonymize(amr):
    os.chdir('NeuralAmrReent/')
    p = subprocess.check_output(["bash", "anonDeAnon_java.sh", "anonymizeAmrFull", "false", amr])
    p = p.decode('utf-8')
    amr = p.split('#')[0]
    alignments = p.split('#')[1]
    os.chdir('..')
    return amr, alignments

def deanonymize(amr, alignments):
    os.chdir('NeuralAmrReent/')
    amr = amr.strip() + '#' + alignments.strip()
    p = subprocess.check_output(["bash", "anonDeAnon_java.sh", "deAnonymizeText", "false", amr])
    text = p.decode('utf-8')
    os.chdir('..')
    return text

def translate(opt):
    translator = build_translator(opt, report_score=False)
    translator.translate(src_path=opt.src,
                         tgt_path=opt.tgt,
                         src_dir=opt.src_dir,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug, demo=True)

def run(amr, model):
    amr = amr.replace('\n', ' ').strip()
    try:
        amr, alignments = anonymize(amr)
    except:
        return 'Anonymization with github.com/sinantie/NeuralAmr returned an error'

    if model == 'seq':
        model_path = 'models/seq_cpu/amr-model_step_10000.pt'
    elif model == 'tree':
        model_path = 'models/tree_gcn_seq_cpu/amr-model_step_15000.pt'
    else:
        model_path = 'models/graph_gcn_seq_cpu/amr-model_step_12500.pt'

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
    opt.replace_unk = True
    opt.max_length = 125
    try:
        translate(opt)
    except:
        return 'The generation model returned an error'

    out = open(tmp_file_path + '.pred', 'r').read()
    try:
        out = deanonymize(out, alignments)
    except:
        return 'De-anonymization with github.com/sinantie/NeuralAm returned an error'
   
    os.remove(tmp_file_path)
    os.remove(tmp_file_path + '.pred')
    return out

#print(run('(e / eat-01)', 'graph'))
#print(run('(a2 / and :op1 (g / go-06 :mode imperative :ARG0 (y / you) :ARG2 (a / ahead)) :op2 (p / process-01 :mode imperative :ARG0 y :ARG1 (t / that) :duration (w / while)))', 'graph'))
