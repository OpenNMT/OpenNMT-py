from __future__ import division

import onmt
import torch
import argparse

import onmt.Models

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-output_dir', default='.',
                    help="""Path to output the embeddings""")
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def write_embeddings(filename, dict, embeddings):
    with open(filename, 'w') as file:
        for i in range(len(embeddings)):
            str = dict.idxToLabel[i].encode("utf-8")
            for j in range(len(embeddings[0])):
                str = str + " %5f" % (embeddings[i][j])
            file.write(str + "\n")


def main():
    opt = parser.parse_args()
    checkpoint = torch.load(opt.model)
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    model_opt = checkpoint['opt']
    src_dict = checkpoint['dicts']['src']
    tgt_dict = checkpoint['dicts']['tgt']

    encoder = onmt.Models.Encoder(model_opt, src_dict)
    decoder = onmt.Models.Decoder(model_opt, tgt_dict)
    encoder_embeddings = encoder.word_lut.weight.data.tolist()
    decoder_embeddings = decoder.word_lut.weight.data.tolist()

    print("Writing source embeddings")
    write_embeddings(opt.output_dir + "/src_embeddings.txt", src_dict,
                     encoder_embeddings)

    print("Writing target embeddings")
    write_embeddings(opt.output_dir + "/tgt_embeddings.txt", tgt_dict,
                     decoder_embeddings)

    print('... done.')
    print('Converting model...')


if __name__ == "__main__":
    main()
