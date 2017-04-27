from __future__ import division

import onmt
import torch
import argparse
import math

from torch.autograd import Variable

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src', required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size', type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had the highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
# parser.add_argument('-phrase_table',
#                     help="""Path to source-target dictionary to replace UNK
#                     tokens. See README.md for the format of this file.""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")

parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal / wordsTotal)))


def addone(f):
    for line in f:
        yield line
    yield None


def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = onmt.Translator(opt)

    outF = open(opt.output, 'w')

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    srcBatch, tgtBatch = [], []

    count = 0

    beamSize = opt.beam_size

    tgtF = open(opt.tgt) if opt.tgt else None
    for line in addone(open(opt.src)):
        srcTokens = line.split()
        # input_Variable = self.src_dict.convertToIdx(b, onmt.Constants.UNK_WORD)
        encoder_indexes = translator.src_dict.convertToIdx(srcTokens, onmt.Constants.UNK_WORD)
        encoder_variable = Variable(torch.LongTensor(encoder_indexes).view(-1, 1))  # seq_len * 1

        # (1) run the encoder on the src
        encStates, context = translator.model.encoder(encoder_variable)
        # srcBatch:   seq_len * batchSize: data, batchSize: length of data;
        # encStates(hidden_t):    h_t, c_t shape is (num_layers * num_directions, batch, hidden_size)
        # context:  seq_len * batchSize * (hiddenSize * num_directions):

        rnnSize = context.size(2)  # hiddenSize * num_directions(1 or 2_brnn) eg: 500 * 1

        # reshape
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        encStates = (translator.model._fix_enc_hidden(encStates[0]),
                     translator.model._fix_enc_hidden(encStates[1]))

        # Expand tensors for each beam.
        context = Variable(context.data.repeat(1, beamSize, 1))
        # context: encoding of source sentence, (max_sentence_length, beamSize * batchSize, hiddenSize)

        init_decStates = (Variable(encStates[0].data.repeat(1, beamSize, 1)),
                     Variable(encStates[1].data.repeat(1, beamSize, 1)))
        # decStates: h_0, c_0 (layer_num, beamSize * batchSize, hiddenSize)

        init_decOut = translator.model.make_init_decoder_output(context)
        # decOut: (beamSize * batchSize, hiddenSize)

        max_breath = 100
        beam_stack = []
        beam_lower = 0
        beam_upper = 1000
        beam_stack.append([beam_lower, beam_upper])

        allHyp, allScores, allAttn = [], [], []
        n_best = opt.n_best

        for mb in range(max_breath):
            beam = onmt.BeamStack(beamSize, opt.cuda)
            decStates = init_decStates
            decOut = init_decOut
            for i in range(opt.max_sent_length):
                input = torch.stack([beam.getCurrentState()]).t().contiguous().view(1, -1)

                decOut, decStates, attn = translator.model.decoder(
                    Variable(input, volatile=True), decStates, context, decOut)

                decOut = decOut.squeeze(0)
                out = translator.model.generator.forward(decOut)

                wordLk = out.view(beamSize, 1, -1).transpose(0, 1).contiguous()
                attn = attn.view(beamSize, 1, -1).transpose(0, 1).contiguous()

                done, score = beam.advance(wordLk.data[0], attn.data[0], beam_stack, beam_lower, beam_upper, i)
                if done:
                    # package everything up

                    scores, ks = beam.sortBest()
                    allScores += [scores[:n_best]]
                    hyps, attn = zip(*[beam.getHyp(k) for k in ks[:n_best]])
                    allHyp += [hyps]

                    beam_upper = score
                    break

            while beam_stack[-1] and beam_stack[-1][1] >= beam_upper:
                beam_stack.pop()
            if len(beam_stack) == 0:
                break
            beam_stack[-1][0] = beam_stack[-1][1]
            beam_stack[-1][1] = beam_upper

        outF.write(" ".join(allHyp[-1]) + "\n")
        outF.flush()

if __name__ == "__main__":
    main()
