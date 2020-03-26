#!/usr/bin/env python
import argparse
import torch

def combine(enc_path, dec_path, out_path):
    enc = torch.load(enc_path)
    dec = torch.load(dec_path)
    for key, val in enc['model'].items():
        if key.startswith('encoder'):
            dec['model'][key] = val
    torch.save(dec, out_path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('encoder')
    parser.add_argument('decoder')
    parser.add_argument('combined_output')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    combine(args.encoder, args.decoder, args.combined_output)

if __name__ == '__main__':
    main()
