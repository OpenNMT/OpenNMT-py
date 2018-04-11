#!/usr/bin/env python
from __future__ import division, unicode_literals
import argparse

from onmt.translate.Translator import make_translator

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import onmt.opts

            if opt.attn_debug:
                srcs = trans.src_raw
                preds = trans.pred_sents[0]
                preds.append('</s>')
                attns = trans.attns[0].tolist()
                header_format = "{:>10.10} " + "{:>10.7} " * len(srcs)
                row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                output = header_format.format("", *trans.src_raw) + '\n'
                for word, row in zip(preds, attns):
                    max_index = row.index(max(row))
                    row_format = row_format.replace(
                        "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
                    row_format = row_format.replace(
                        "{:*>10.7f} ", "{:>10.7f} ", max_index)
                    output += row_format.format(word, *row) + '\n'
                    row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                os.write(1, output.encode('utf-8'))


def main(opt):
    translator = make_translator(opt, report_score=True)
    translator.translate(opt.src_dir, opt.src, opt.tgt,
                         opt.batch_size, opt.attn_debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    main(opt)
