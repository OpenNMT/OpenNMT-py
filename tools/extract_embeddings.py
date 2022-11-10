import argparse

import torch
import onmt
import onmt.model_builder

from onmt.utils.parse import ArgumentParser
import onmt.opts
from onmt.inputters.inputter import dict_to_vocabs
from onmt.utils.misc import use_gpu
from onmt.utils.logging import init_logger, logger

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-output_dir', default='.',
                    help="""Path to output the embeddings""")
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def write_embeddings(filename, vocab, embeddings):
    with open(filename, 'wb') as file:
        for i in range(min(len(embeddings), len(vocab))):
            str = vocab.lookup_index(i).encode("utf-8")
            for j in range(len(embeddings[0])):
                str = str + (" %5f" % (embeddings[i][j])).encode("utf-8")
            file.write(str + b"\n")


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    onmt.opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # Add in default model arguments, possibly added since training.
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    model_opt = checkpoint['opt']

    vocabs = dict_to_vocabs(checkpoint['vocab'])
    src_vocab = vocabs['src']  # assumes src is text
    tgt_vocab = vocabs['tgt']

    model_opt = checkpoint['opt']
    # this patch is no longer needed included in converter
    # if hasattr(model_opt, 'rnn_size'):
    #     model_opt.hidden_size = model_opt.rnn_size
    for arg in dummy_opt.__dict__:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt.__dict__[arg]

    # build_base_model expects updated and validated opts
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)

    model = onmt.model_builder.build_base_model(
        model_opt, vocabs, use_gpu(opt), checkpoint)
    encoder = model.encoder  # no encoder for LM task
    decoder = model.decoder

    encoder_embeddings = encoder.embeddings.word_lut.weight.data.tolist()
    decoder_embeddings = decoder.embeddings.word_lut.weight.data.tolist()

    logger.info("Writing source embeddings")
    write_embeddings(opt.output_dir + "/src_embeddings.txt", src_vocab,
                     encoder_embeddings)

    logger.info("Writing target embeddings")
    write_embeddings(opt.output_dir + "/tgt_embeddings.txt", tgt_vocab,
                     decoder_embeddings)

    logger.info('... done.')
    logger.info('Converting model...')


if __name__ == "__main__":
    init_logger('extract_embeddings.log')
    main()
