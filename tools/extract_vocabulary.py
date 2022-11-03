#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, required=True)
    parser.add_argument("-out_file", type=str, required=True)
    parser.add_argument("-side", choices=['src', 'tgt'], help="""Specifies
                               'src' or 'tgt' side for 'vocab' file_type.""")

    opt = parser.parse_args()

    if opt.side not in ['src', 'tgt']:
        raise ValueError("If using -file_type='vocab', specifies "
                         "'src' or 'tgt' argument for -side.")
    import torch
    print("Reading input file...")
    model = torch.load(opt.model, map_location=torch.device("cpu"))
    voc = model['vocab'][opt.side]

    print("Writing vocabulary file...")
    with open(opt.out_file, "wb") as f:
        for w in voc:
            f.write(u"{0}\n".format(w).encode("utf-8"))


if __name__ == "__main__":
    main()
