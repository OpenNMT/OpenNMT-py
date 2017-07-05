import onmt
import onmt.Models
import onmt.modules
import onmt.IO
import torch.nn as nn
import torch
from torch.autograd import Variable


class Translator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.beam_accum = None

        checkpoint = torch.load(opt.model,
                                map_location=lambda storage, loc: storage)

        model_opt = checkpoint['opt']
        self.src_dict = checkpoint['dicts']['src']
        self.tgt_dict = checkpoint['dicts']['tgt']
        self.align = self.src_dict.align(self.tgt_dict)
        self.src_feature_dicts = checkpoint['dicts'].get('src_features', None)
        self._type = model_opt.encoder_type \
            if "encoder_type" in model_opt else "text"

        self.copy_attn = model_opt.copy_attn \
            if "copy_attn" in model_opt else "std"

        if self._type == "text":
            encoder = onmt.Models.Encoder(model_opt, self.src_dict,
                                          self.src_feature_dicts)
        elif self._type == "img":
            encoder = onmt.modules.ImageEncoder(model_opt)

        decoder = onmt.Models.Decoder(model_opt, self.tgt_dict)
        model = onmt.Models.NMTModel(encoder, decoder)

        if not self.copy_attn or self.copy_attn == "std":
            generator = nn.Sequential(
                nn.Linear(model_opt.rnn_size, self.tgt_dict.size()),
                nn.LogSoftmax())
        elif self.copy_attn:
            generator = onmt.modules.CopyGenerator(model_opt, self.src_dict,
                                                   self.tgt_dict)

        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])

        if opt.cuda:
            model.cuda()
            generator.cuda()
        else:
            model.cpu()
            generator.cpu()

        model.generator = generator

        self.model = model
        self.model.eval()

    def initBeamAccum(self):
        self.beam_accum = {
            "predicted_ids": [],
            "beam_parent_ids": [],
            "scores": [],
            "log_probs": []}

    def buildData(self, srcBatch, goldBatch):
        srcFeats = []
        if self.src_feature_dicts:
            srcFeats = [[] for i in range(len(self.src_feature_dicts))]
        srcData = []
        tgtData = None
        for b in srcBatch:
            _, srcD, srcFeat = onmt.IO.readSrcLine(b, self.src_dict,
                                                   self.src_feature_dicts,
                                                   self._type)
            srcData += [srcD]
            for i in range(len(srcFeats)):
                srcFeats[i] += [srcFeat[i]]

        if goldBatch:
            for b in goldBatch:
                _, tgtD, tgtFeat = onmt.IO.readTgtLine(b, self.src_dict,
                                                       None, self._type)
                tgtData += [tgtD]

        return onmt.Dataset(srcData, tgtData, self.opt.batch_size,
                            self.opt.cuda, volatile=True,
                            data_type=self._type,
                            srcFeatures=srcFeats)

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == onmt.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def translateBatch(self, batch):
        beamSize = self.opt.beam_size
        batchSize = batch.batchSize

        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(batch.src)
        encStates = self.model.init_decoder_state(context, encStates)

        decoder = self.model.decoder
        attentionLayer = decoder.attn
        useMasking = (self._type == "text")

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        padMask = None
        if useMasking:
            padMask = batch.words().data.eq(onmt.Constants.PAD).t()

        def mask(padMask):
            if useMasking:
                attentionLayer.applyMask(padMask)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = context.data.new(batchSize).zero_()
        if batch.tgt is not None:
            decStates = encStates
            mask(padMask)
            decOut, decStates, attn = decoder(batch.tgt[:-1],
                                              context, decStates)
            for dec_t, tgt_t in zip(decOut, batch.tgt[1:].data):
                gen_t = self.model.generator.forward(dec_t)
                tgt_t = tgt_t.unsqueeze(1)
                scores = gen_t.data.gather(1, tgt_t)
                scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
                goldScores += scores

        #  (3) run the decoder to generate sentences, using beam search
        # Each hypothesis in the beam uses the same context
        # and initial decoder state
        context = Variable(context.data.repeat(1, beamSize, 1))
        batch_src = Variable(batch.src.data.repeat(1, beamSize, 1))
        decStates = encStates
        decStates.repeatBeam_(beamSize)
        beam = [onmt.Beam(beamSize, self.opt.cuda)
                for _ in range(batchSize)]
        if useMasking:
            padMask = batch.src.data[:, :, 0].eq(
                onmt.Constants.PAD).t() \
                                   .unsqueeze(0) \
                                   .repeat(beamSize, 1, 1)

        #  (3b) The main loop
        for i in range(self.opt.max_sent_length):
            # (a) Run RNN decoder forward one step.
            mask(padMask)
            input = torch.stack([b.getCurrentState() for b in beam])\
                         .t().contiguous().view(1, -1)
            input = Variable(input, volatile=True)
            decOut, decStates, attn = self.model.decoder(input, batch_src,
                                                         context, decStates)
            decOut = decOut.squeeze(0)
            # decOut: (beam*batch) x numWords
            attn["std"] = attn["std"].view(beamSize, batchSize, -1) \
                                     .transpose(0, 1).contiguous()

            # (b) Compute a vector of batch*beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(decOut)
            else:
                # Copy Attention Case
                words = batch.words().t()
                words = torch.stack([words[i] for i, b in enumerate(beam)])\
                             .contiguous()
                attn_copy = attn["copy"].view(beamSize, batchSize, -1) \
                                        .transpose(0, 1).contiguous()

                out, c_attn_t \
                    = self.model.generator.forward(
                        decOut, attn_copy.view(-1, batch_src.size(0)))

                for b in range(out.size(0)):
                    for c in range(c_attn_t.size(1)):
                        v = self.align[words[0, c].data[0]]
                        if v != onmt.Constants.PAD:
                            out[b, v] += c_attn_t[b, c]
                out = out.log()

            word_scores = out.view(beamSize, batchSize, -1) \
                .transpose(0, 1).contiguous()
            # batch x beam x numWords

            # (c) Advance each beam.
            active = []
            for b in range(batchSize):
                is_done = beam[b].advance(word_scores.data[b],
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
                valid_attn = batch.src.data[:, b, 0].ne(onmt.Constants.PAD) \
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

    def translate(self, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        batch = dataset[0]
        batchSize = batch.batchSize

        #  (2) translate
        pred, predScore, attn, goldScore = self.translateBatch(batch)
        pred, predScore, attn, goldScore = list(zip(
            *sorted(zip(pred, predScore, attn, goldScore, batch.indices),
                    key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(batchSize):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                 for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, goldScore, attn, batch.src
