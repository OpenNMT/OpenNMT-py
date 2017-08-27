import onmt
import onmt.Models
import onmt.modules
import onmt.IO
import torch
from torch.autograd import Variable


class Translator(object):
    def __init__(self, opt, dummy_opt={}):
        # Add in default model arguments, possibly added since training.
        self.opt = opt
        checkpoint = torch.load(opt.model,
                                map_location=lambda storage, loc: storage)
        self.fields = onmt.IO.ONMTDataset.load_fields(checkpoint['vocab'])

        model_opt = checkpoint['opt']
        for arg in dummy_opt:
            if arg not in model_opt:
                model_opt.__dict__[arg] = dummy_opt[arg]

        self._type = model_opt.model_type
        self.copy_attn = model_opt.copy_attn

        self.model = onmt.Models.make_base_model(opt, model_opt, self.fields,
                                                 opt.cuda, checkpoint)
        self.model.eval()

        # for debugging
        self.beam_accum = None

    def initBeamAccum(self):
        self.beam_accum = {
            "predicted_ids": [],
            "beam_parent_ids": [],
            "scores": [],
            "log_probs": []}

    def buildTargetTokens(self, pred, src, attn, copy_vocab):
        vocab = self.fields["tgt"].vocab
        tgt_eos = vocab.stoi[onmt.IO.EOS_WORD]
        tokens = []
        for tok in pred:
            if tok != tgt_eos:
                if tok < len(vocab):
                    tokens.append(vocab.itos[tok])
                else:
                    tokens.append(copy_vocab.itos[tok - len(vocab)])
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == onmt.IO.UNK:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def translateBatch(self, batch, data):
        beamSize = self.opt.beam_size
        batchSize = batch.batch_size
        _, src_lengths = batch.src
        src = onmt.IO.make_features(batch, self.fields)

        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(src, lengths=src_lengths)
        encStates = self.model.init_decoder_state(context, encStates)

        useMasking = (self._type == "text")
        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        padMask = None
        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.IO.PAD_WORD]
        if useMasking:
            pad = self.fields["src"].vocab.stoi[onmt.IO.PAD_WORD]
            padMask = src[:, :, 0].data.eq(pad).t()

        def mask(padMask):
            if useMasking:
                self.model.decoder.attn.applyMask(padMask)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = context.data.new(batchSize).zero_()
        if "tgt" in batch.__dict__:
            decStates = encStates
            mask(padMask.unsqueeze(0))
            decOut, decStates, attn = self.model.decoder(batch.tgt[:-1],
                                                         batch.src,
                                                         context,
                                                         decStates)
            for dec_t, tgt_t in zip(decOut, batch.tgt[1:].data):
                gen_t = self.model.generator.forward(dec_t)
                tgt_t = tgt_t.unsqueeze(1)
                scores = gen_t.data.gather(1, tgt_t)
                scores.masked_fill_(tgt_t.eq(tgt_pad), 0)
                goldScores += scores

        #  (3) run the decoder to generate sentences, using beam search
        # Each hypothesis in the beam uses the same context
        # and initial decoder state
        context = Variable(context.data.repeat(1, beamSize, 1), volatile=True)
        batch_src = Variable(src.data.repeat(1, beamSize, 1), volatile=True)
        batch_src_map = Variable(batch.src_map.data.repeat(1, beamSize, 1),
                                 volatile=True)
        decStates = encStates
        decStates.repeatBeam_(beamSize)
        beam = [onmt.Beam(beamSize, cuda=self.opt.cuda,
                          vocab=self.fields["tgt"].vocab)
                for __ in range(batchSize)]
        if useMasking:
            padMask = src.data[:, :, 0].eq(pad).t() \
                                               .unsqueeze(0) \
                                               .repeat(beamSize, 1, 1)

        #  (3b) The main loop
        for i in range(self.opt.max_sent_length):
            # (a) Run RNN decoder forward one step.
            mask(padMask)
            input = torch.stack([b.getCurrentState() for b in beam])\
                         .t().contiguous().view(1, -1)
            input.masked_fill_(input.gt(len(self.fields["tgt"].vocab)-1), 0)
            input = Variable(input, volatile=True)
            decOut, decStates, attn = self.model.decoder(input, batch_src,
                                                         context, decStates)
            # print(decStates.all[0][:, 0, 0])
            decOut = decOut.squeeze(0)
            # decOut: (beam*batch) x numWords
            attn["std"] = attn["std"].view(beamSize, batchSize, -1) \
                                     .transpose(0, 1).contiguous()

            # (b) Compute a vector of batch*beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(decOut).data
            else:
                # print(attn["copy"].size())
                attn_copy = attn["copy"].view(beamSize, batchSize, -1) \
                                        .transpose(0, 1).contiguous()
                out = self.model.generator.forward(
                    decOut, attn_copy.view(-1, batch_src.size(0)),
                    batch_src_map)
                out = data.collapseCopyScores(
                    out.data.view(batchSize, beamSize, -1).transpose(0, 1),
                    batch, self.fields["tgt"].vocab)
                out = out.log().transpose(0, 1).contiguous()\
                                               .view(beamSize * batchSize, -1)

            word_scores = out.view(beamSize, batchSize, -1) \
                .transpose(0, 1).contiguous()
            # batch x beam x numWords

            # (c) Advance each beam.
            active = []
            for b in range(batchSize):
                is_done = beam[b].advance(word_scores[b],
                                          attn["std"].data[b])
                if not is_done:
                    active += [b]
                decStates.beamUpdate_(b, beam[b].getCurrentOrigin(),
                                      beamSize)
            if not active:
                break

        #  (4) package everything up
        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            hyps, attn = [], []
            for k in ks[:n_best]:
                hyp, att = beam[b].getHyp(k)
                hyps.append(hyp)
                attn.append(att)
            allHyp += [hyps]
            if useMasking:
                valid_attn = src.data[:, b, 0].ne(pad) \
                                              .nonzero().squeeze(1)
                attn = [a.index_select(1, valid_attn) for a in attn]
            allAttn += [attn]

            # For debugging visualization.
            if self.beam_accum:
                self.beam_accum["beam_parent_ids"].append(
                    [t.tolist()
                     for t in beam[b].prevKs])
                self.beam_accum["scores"].append([
                    ["%4f" % s for s in t.tolist()]
                    for t in beam[b].allScores][1:])
                self.beam_accum["predicted_ids"].append(
                    [[self.tgt_dict.getLabel(id)
                      for id in t.tolist()]
                     for t in beam[b].nextYs][1:])

        return allHyp, allScores, allAttn, goldScores

    def translate(self, batch, data):
        #  (1) convert words to indexes
        batchSize = batch.batch_size

        #  (2) translate
        pred, predScore, attn, goldScore = self.translateBatch(batch, data)
        pred, predScore, attn, goldScore, i = list(zip(
            *sorted(zip(pred, predScore, attn, goldScore,
                        batch.indices.data),
                    key=lambda x: x[-1])))
        inds, perm = torch.sort(batch.indices.data)

        #  (3) convert indexes to words
        predBatch = []
        src = batch.src[0].data.index_select(1, perm)
        for b in range(batchSize):
            src_vocab = data.src_vocabs[inds[b]]
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], src[:, b],
                                        attn[b][n], src_vocab)
                 for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, goldScore, attn, src
