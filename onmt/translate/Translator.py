import torch
from torch.autograd import Variable

import onmt.translate.Beam
import onmt.io


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 cuda=False,
                 beam_trace=False,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[]):
        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate_batch(self, batch, data):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object


        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])

        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src

        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
                                        src, memory_bank, enc_states)

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data)\
                                                  .long()\
                                                  .fill_(memory_bank.size(0))

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None
        memory_bank = rvar(memory_bank.data)
        memory_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
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
            dec_out, dec_states, attn = self.model.decoder(
                inp, memory_bank, dec_states, memory_lengths=memory_lengths)
            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
                beam_attn = unbottle(attn["std"])
            else:
                out = self.model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()
                beam_attn = unbottle(attn["copy"])
            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j],
                          beam_attn.data[:, j, :memory_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = \
            self.model.decoder.init_decoder_state(src, memory_bank, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        dec_out, _, _ = self.model.decoder(
            tgt_in, memory_bank, dec_states, memory_lengths=src_lengths)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores
