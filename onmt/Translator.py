import torch
from torch.autograd import Variable

import onmt
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
import onmt.IO
from onmt.Utils import use_gpu


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

        self.model = onmt.ModelConstructor.make_base_model(
                            model_opt, self.fields, use_gpu(opt), checkpoint)
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
        tokens = []
        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                tokens.append(copy_vocab.itos[tok - len(vocab)])
            if tokens[-1] == onmt.IO.EOS_WORD:
                tokens = tokens[:-1]
                break

        if self.opt.replace_unk and attn is not None:
            for i in range(len(tokens)):
                if tokens[i] == vocab.itos[onmt.IO.UNK]:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = self.fields["src"].vocab.itos[src[maxIndex[0]]]
        return tokens

    def _runTarget(self, batch, data):

        _, src_lengths = batch.src
        src = onmt.IO.make_features(batch, 'src')
        tgt_in = onmt.IO.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(src, src_lengths)
        decStates = self.model.decoder.init_decoder_state(
                                        src, context, encStates)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.opt.cuda else torch
        goldScores = tt.FloatTensor(batch.batch_size).fill_(0)
        decOut, decStates, attn = self.model.decoder(
            tgt_in, context, decStates)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.IO.PAD_WORD]
        for dec, tgt in zip(decOut, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            goldScores += scores
        return goldScores

    def translateBatch(self, batch, dataset):
        beam_size = self.opt.beam_size
        batch_size = batch.batch_size

        # (1) Run the encoder on the src.
        _, src_lengths = batch.src
        src = onmt.IO.make_features(batch, 'src')
        encStates, context = self.model.encoder(src, src_lengths)
        decStates = self.model.decoder.init_decoder_state(
                                        src, context, encStates)

        #  (1b) Initialize for the decoder.
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        # Repeat everything beam_size times.
        context = rvar(context.data)
        src = rvar(src.data)
        srcMap = rvar(batch.src_map.data)
        decStates.repeat_beam_size_times(beam_size)
        scorer = None
        # scorer=onmt.GNMTGlobalScorer(0.3, 0.4)
        beam = [onmt.Beam(beam_size, n_best=self.opt.n_best,
                          cuda=self.opt.cuda,
                          vocab=self.fields["tgt"].vocab,
                          global_scorer=scorer)
                for __ in range(batch_size)]

        # (2) run the decoder to generate sentences, using beam search.

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        for i in range(self.opt.max_sent_length):

            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            decOut, decStates, attn = \
                self.model.decoder(inp, context, decStates)
            decOut = decOut.squeeze(0)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(decOut).data
                out = unbottle(out)
                # beam x tgt_vocab
            else:
                out = self.model.generator.forward(decOut,
                                                   attn["copy"].squeeze(0),
                                                   srcMap)
                # beam x (tgt_vocab + extra_vocab)
                out = dataset.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab)
                # beam x tgt_vocab
                out = out.log()

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j],  unbottle(attn["std"]).data[:, j])
                decStates.beam_update(j, b.getCurrentOrigin(), beam_size)

        if "tgt" in batch.__dict__:
            allGold = self._runTarget(batch, dataset)
        else:
            allGold = [0] * batch_size

        # (3) Package everything up.
        allHyps, allScores, allAttn = [], [], []
        for b in beam:
            n_best = self.opt.n_best
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            allHyps.append(hyps)
            allScores.append(scores)
            allAttn.append(attn)

        return allHyps, allScores, allAttn, allGold

    def translate(self, batch, data):
        #  (1) convert words to indexes
        batch_size = batch.batch_size

        #  (2) translate
        pred, predScore, attn, goldScore = self.translateBatch(batch, data)
        assert(len(goldScore) == len(pred))
        pred, predScore, attn, goldScore, i = list(zip(
            *sorted(zip(pred, predScore, attn, goldScore,
                        batch.indices.data),
                    key=lambda x: x[-1])))
        inds, perm = torch.sort(batch.indices.data)

        #  (3) convert indexes to words
        predBatch, goldBatch = [], []
        src = batch.src[0].data.index_select(1, perm)
        if self.opt.tgt:
            tgt = batch.tgt.data.index_select(1, perm)
        for b in range(batch_size):
            src_vocab = data.src_vocabs[inds[b]]
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], src[:, b],
                                        attn[b][n], src_vocab)
                 for n in range(self.opt.n_best)])
            if self.opt.tgt:
                goldBatch.append(
                    self.buildTargetTokens(tgt[1:, b], src[:, b],
                                           None, None))
        return predBatch, goldBatch, predScore, goldScore, attn, src
