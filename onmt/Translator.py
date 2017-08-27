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

        self._type = model_opt.encoder_type
        self.copy_attn = model_opt.copy_attn

        self.model = onmt.Models.make_base_model(opt, model_opt, self.fields,
                                                 opt.cuda, checkpoint)
        self.model.eval()
        self.model.generator.eval()

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

    def _runTarget(self, batch, data):

        _, src_lengths = batch.src
        src = onmt.IO.make_features(batch, self.fields)

        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(src, src_lengths)
        decStates = self.model.init_decoder_state(context, encStates)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = 0
        decOut, decStates, attn = self.model.decoder(
            batch.tgt[:-1], batch.src, context, decStates)

        aeq(decOut.size(), batch.tgt[1:].data.size())
        for dec, tgt in zip(decOut, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            goldScores += scores[0]
        return goldScores


    def translateBatch(self, batch, dataset):
        beamSize = self.opt.beam_size
        batchSize = batch.batch_size

        #  (1) run the encoder on the src
        _, src_lengths = batch.src
        src = make_features(batch, self.fields)
        encStates, context = self.model.encoder(src, src_lengths)
        decStates = self.model.init_decoder_state(context, encStates)

        #  (1b) initialize for the decoder.
        def var(a): return Variable(a, volatile=True)
        def rvar(a): return var(a.repeat(1, beamSize, 1))

        # Repeat everything beam_times
        context = rvar(context.data)
        src = rvar(src.data)
        srcMap = rvar(batch.src_map.data)
        decStates.repeatBeam_(beamSize)
        beam = [onmt.Beam(beamSize, n_best=self.opt.n_best, cuda=self.opt.cuda,
                          vocab=self.fields["tgt"].vocab)
                for _ in range(batchSize)]

        #  (2) run the decoder to generate sentences, using beam search
        i = 0
        def bottle(m):
            return m.view(batchSize * beamSize, -1)
        def unbottle(m):
            return m.view(beamSize, batchSize, -1)

        while i < self.opt.max_sent_length \
              or any([len(b.finished) == 0 for b in beam]):
            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(1, -1))
            
            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(inp.gt(len(self.fields["tgt"].vocab) - 1), 0)
                            
            # Run one step.
            decOut, decStates, attn = \
                self.model.decoder(inp, src, context, decStates)
            decOut = decOut.squeeze(0)
            # decOut: beam x rnn_size
            
            # (b) Compute a vector of batch*beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(decOut).data
                out = unbottle(out)
                # beam x tgt_vocab
            else:                
                out = self.model.generator.forward(decOut, attn["copy"].squeeze(0),
                                                   srcMap)
                # beam x (tgt_vocab + extra_vocab)
                out = dataset.collapseCopyScores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab)
                # beam x tgt_vocab
                out = out.log()

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                is_done = b.advance(out[:, j], unbottle(attn["copy"]).data[:, j])
                decStates.beamUpdate_(j, b.getCurrentOrigin(), beamSize)
                if is_done:
                    break
            i += 1
            
        if "tgt" in batch.__dict__:
            allGold = self._runTarget(batch, dataset)
        else:
            allGold = [0] * batchSize

            
        #  (3) package everything up
        allHyps, allScores, allAttn = [], [], []
        for b in beam:
            n_best = self.opt.n_best
            scores, ks = b.sortFinished()
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks):# [:n_best]:
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            allHyps.append(hyps)
            allScores.append(scores)
            allAttn.append(attn)
            
        return allHyps, allScores, allAttn, allGold

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
