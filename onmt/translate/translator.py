#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import configargparse
import codecs
import os
import math

import torch

from itertools import count
from onmt.utils.misc import tile

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.opts as opts
import onmt.decoders.ensemble


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    dummy_parser = configargparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    load_test_model = onmt.decoders.ensemble.load_test_model \
        if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt, dummy_opt.__dict__)

    scorer = onmt.translate.GNMTGlobalScorer(opt)

    translator = Translator(
        model,
        fields,
        opt,
        model_opt,
        global_scorer=scorer,
        out_file=out_file,
        report_score=report_score,
        logger=logger
    )
    return translator


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
       logger(logging.Logger): logger.
    """

    def __init__(
        self,
        model,
        fields,
        opt,
        model_opt,
        global_scorer=None,
        out_file=None,
        report_score=True,
        logger=None
    ):

        self.model = model
        self.fields = fields
        self.gpu = opt.gpu
        self.cuda = opt.gpu > -1

        self.n_best = opt.n_best
        self.max_length = opt.max_length
        self.beam_size = opt.beam_size
        self.min_length = opt.min_length
        self.stepwise_penalty = opt.stepwise_penalty
        self.dump_beam = opt.dump_beam
        self.block_ngram_repeat = opt.block_ngram_repeat
        self.ignore_when_blocking = set(opt.ignore_when_blocking)
        self.sample_rate = opt.sample_rate
        self.window_size = opt.window_size
        self.window_stride = opt.window_stride
        self.window = opt.window
        self.image_channel_size = opt.image_channel_size
        self.replace_unk = opt.replace_unk
        self.data_type = opt.data_type
        self.verbose = opt.verbose
        self.report_bleu = opt.report_bleu
        self.report_rouge = opt.report_rouge
        self.fast = opt.fast

        self.copy_attn = model_opt.copy_attn

        self.global_scorer = global_scorer
        self.out_file = out_file
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate(
        self,
        src,
        tgt=None,
        src_dir=None,
        batch_size=None,
        attn_debug=False
    ):
        """
        Translate content of `src_data_iter` (if not None) or `src_path`
        and get gold scores if one of `tgt_data_iter` or `tgt_path` is set.

        Note: batch_size must not be None
        Note: one of ('src_path', 'src_data_iter') must not be None

        Args:
            src_path (str): filepath of source data
            tgt_path (str): filepath of target data or None
            src_dir (str): source directory path
                (used for Audio and Image datasets)
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        assert src is not None

        if batch_size is None:
            raise ValueError("batch_size must be set")

        data = inputters.build_dataset(
            self.fields,
            self.data_type,
            src=src,
            tgt=tgt,
            src_dir=src_dir,
            sample_rate=self.sample_rate,
            window_size=self.window_size,
            window_stride=self.window_stride,
            window=self.window,
            use_filter_pred=self.use_filter_pred,
            image_channel_size=self.image_channel_size,
            dynamic_dict=self.copy_attn
        )

        cur_device = "cuda" if self.cuda else "cpu"

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=cur_device,
            batch_size=batch_size,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

        builder = onmt.translate.TranslationBuilder(
            data, self.fields, self.n_best, self.replace_unk, tgt
        )

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        for batch in data_iter:
            batch_data = self.translate_batch(
                batch, data, attn_debug, fast=self.fast
            )
            translations = builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                all_predictions += [n_best_preds]
                self.out_file.write('\n'.join(n_best_preds) + '\n')
                self.out_file.flush()

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode('utf-8'))

                if attn_debug:
                    preds = trans.pred_sents[0]
                    preds.append('</s>')
                    attns = trans.attns[0].tolist()
                    if self.data_type == 'text':
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(attns[0]))]
                    header_format = "{:>10.10} " + "{:>10.7} " * len(srcs)
                    row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    output = header_format.format("", *srcs) + '\n'
                    for word, row in zip(preds, attns):
                        max_index = row.index(max(row))
                        row_format = row_format.replace(
                            "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
                        row_format = row_format.replace(
                            "{:*>10.7f} ", "{:>10.7f} ", max_index)
                        output += row_format.format(word, *row) + '\n'
                        row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    os.write(1, output.encode('utf-8'))

        if self.report_score:
            msg = self._report_score('PRED', pred_score_total,
                                     pred_words_total)
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
            if tgt is not None:
                msg = self._report_score('GOLD', gold_score_total,
                                         gold_words_total)
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)
                if self.report_bleu:
                    msg = self._report_bleu(tgt)
                    if self.logger:
                        self.logger.info(msg)
                    else:
                        print(msg)
                if self.report_rouge:
                    msg = self._report_rouge(tgt)
                    if self.logger:
                        self.logger.info(msg)
                    else:
                        print(msg)

        if self.dump_beam:
            import json
            json.dump(self.translator.beam_accum,
                      codecs.open(self.dump_beam, 'w', 'utf-8'))
        return all_scores, all_predictions

    def translate_batch(self, batch, data, attn_debug, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            if fast:
                return self._fast_translate_batch(
                    batch,
                    data,
                    self.max_length,
                    min_length=self.min_length,
                    n_best=self.n_best,
                    return_attention=attn_debug or self.replace_unk)
            else:
                return self._translate_batch(batch, data)

    def _run_encoder(self, batch, data_type):
        src = inputters.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src
        elif data_type == 'audio':
            src_lengths = batch.src_lengths
        enc_states, memory_bank, src_lengths = self.model.encoder(
            src, src_lengths)
        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch.batch_size) \
                               .type_as(memory_bank) \
                               .long() \
                               .fill_(memory_bank.size(0))
        return src, enc_states, memory_bank, src_lengths

    def _decode_and_generate(
        self,
        decoder_in,
        memory_bank,
        batch,
        data,
        memory_lengths,
        src_map=None,
        step=None,
        batch_offset=None
    ):

        unk_idx = self.fields["tgt"].vocab.stoi[self.fields["tgt"].unk_token]
        if self.copy_attn:
            # Turn any copied words into UNKs.
            decoder_in = decoder_in.masked_fill(
                decoder_in.gt(len(self.fields["tgt"].vocab) - 1), unk_idx
            )

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder(
            decoder_in, memory_bank, memory_lengths=memory_lengths, step=step
        )

        # Generator forward.
        if not self.copy_attn:
            attn = dec_attn["std"]
            log_probs = self.model.generator(dec_out.squeeze(0))
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        else:
            attn = dec_attn["copy"]
            scores = self.model.generator(dec_out.view(-1, dec_out.size(2)),
                                          attn.view(-1, attn.size(2)),
                                          src_map)
            # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
            if batch_offset is None:
                scores = scores.view(batch.batch_size, -1, scores.size(-1))
            else:
                scores = scores.view(-1, self.beam_size, scores.size(-1))
            scores = data.collapse_copy_scores(
                scores,
                batch,
                self.fields["tgt"].vocab,
                data.src_vocabs,
                batch_dim=0,
                batch_offset=batch_offset
            )
            scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
            log_probs = scores.squeeze(0).log()
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence

        return log_probs, attn

    def _fast_translate_batch(
        self,
        batch,
        data,
        max_length,
        min_length=0,
        n_best=1,
        return_attention=False
    ):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam
        assert not self.use_filter_pred
        assert self.block_ngram_repeat == 0
        assert self.global_scorer.beta == 0

        beam_size = self.beam_size
        batch_size = batch.batch_size
        vocab = self.fields["tgt"].vocab
        start_token = vocab.stoi[self.fields["tgt"].init_token]
        end_token = vocab.stoi[self.fields["tgt"].eos_token]

        # Encoder forward.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(
            batch, data.data_type)
        self.model.decoder.init_state(src, memory_bank, enc_states)

        use_src_map = data.data_type == 'text' and self.copy_attn

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["attention"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["batch"] = batch
        if "tgt" in batch.__dict__:
            results["gold_score"] = self._score_target(
                batch,
                memory_bank,
                src_lengths,
                data,
                batch.src_map if use_src_map else None
            )
            self.model.decoder.init_state(src, memory_bank, enc_states)
        else:
            results["gold_score"] = [0] * batch_size

        # Tile states and memory beam_size times.
        self.model.decoder.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))
        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, beam_size, dim=1) for x in memory_bank)
            mb_device = memory_bank[0].device
        else:
            memory_bank = tile(memory_bank, beam_size, dim=1)
            mb_device = memory_bank.device

        memory_lengths = tile(src_lengths, beam_size)
        src_map = (tile(batch.src_map, beam_size, dim=1)
                   if use_src_map else None)

        top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        batch_offset = torch.arange(batch_size, dtype=torch.long)
        beam_offset = torch.arange(
            0, batch_size * beam_size, step=beam_size, dtype=torch.long,
            device=mb_device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1], start_token, dtype=torch.long,
            device=mb_device)
        alive_attn = None

        # Give full probability to the first beam on the first step.
        topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (beam_size - 1), device=mb_device
        ).repeat(batch_size)

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1, 1)

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank,
                batch,
                data,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=batch_offset
            )

            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty
            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)
            if return_attention:
                current_attn = attn.index_select(1, select_indices)
                if alive_attn is None:
                    alive_attn = current_attn
                else:
                    alive_attn = alive_attn.index_select(1, select_indices)
                    alive_attn = torch.cat([alive_attn, current_attn], 0)

            is_finished = topk_ids.eq(end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)

            # Save finished hypotheses.
            if is_finished.any():
                # Penalize beams that finished.
                topk_log_probs.masked_fill_(is_finished, -1e10)
                is_finished = is_finished.to('cpu')
                top_beam_finished |= is_finished[:, 0].eq(1)
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                attention = (
                    alive_attn.view(
                        alive_attn.size(0), -1, beam_size, alive_attn.size(-1))
                    if alive_attn is not None else None)
                non_finished_batch = []
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:],  # Ignore start_token.
                            attention[:, i, j, :memory_lengths[i]]
                            if attention is not None else None))
                    # End condition is the top beam finished and we can return
                    # n_best hypotheses.
                    if top_beam_finished[i] and len(hypotheses[b]) >= n_best:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred, attn) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                            results["attention"][b].append(
                                attn if attn is not None else [])
                    else:
                        non_finished_batch.append(i)
                non_finished = torch.tensor(non_finished_batch)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                top_beam_finished = top_beam_finished.index_select(
                    0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                non_finished = non_finished.to(topk_ids.device)
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                select_indices = batch_index.view(-1)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
                if alive_attn is not None:
                    alive_attn = attention.index_select(1, non_finished) \
                        .view(alive_attn.size(0),
                              -1, alive_attn.size(-1))

            # Reorder states.
            if isinstance(memory_bank, tuple):
                memory_bank = tuple(x.index_select(1, select_indices)
                                    for x in memory_bank)
            else:
                memory_bank = memory_bank.index_select(1, select_indices)

            memory_lengths = memory_lengths.index_select(0, select_indices)
            self.model.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))
            if src_map is not None:
                src_map = src_map.index_select(1, select_indices)

        return results

    def _translate_batch(self, batch, data):
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab

        # Define a set of tokens to exclude from ngram-blocking
        exclusion_tokens = {vocab.stoi[t] for t in self.ignore_when_blocking}

        pad = vocab.stoi[self.fields['tgt'].pad_token]
        eos = vocab.stoi[self.fields['tgt'].eos_token]
        bos = vocab.stoi[self.fields['tgt'].init_token]
        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=pad, eos=eos, bos=bos,
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(
            batch, data_type)
        self.model.decoder.init_state(src, memory_bank, enc_states)

        results = {}
        results["predictions"] = []
        results["scores"] = []
        results["attention"] = []
        results["batch"] = batch
        if "tgt" in batch.__dict__:
            results["gold_score"] = self._score_target(
                batch, memory_bank, src_lengths, data, batch.src_map
                if data_type == 'text' and self.copy_attn else None)
            self.model.decoder.init_state(src, memory_bank, enc_states)
        else:
            results["gold_score"] = [0] * batch_size

        # (2) Repeat src objects `beam_size` times.
        # We use now  batch_size x beam_size (same as fast mode)
        src_map = (tile(batch.src_map, beam_size, dim=1)
                   if data.data_type == 'text' and self.copy_attn else None)
        self.model.decoder.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, beam_size, dim=1) for x in memory_bank)
        else:
            memory_bank = tile(memory_bank, beam_size, dim=1)
        memory_lengths = tile(src_lengths, beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # (a) Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.

            inp = torch.stack([b.get_current_state() for b in beam])
            inp = inp.view(1, -1, 1)

            # (b) Decode and forward
            out, beam_attn = self._decode_and_generate(
                inp, memory_bank, batch, data, memory_lengths=memory_lengths,
                src_map=src_map, step=i
            )
            out = out.view(batch_size, beam_size, -1)
            beam_attn = beam_attn.view(batch_size, beam_size, -1)

            # (c) Advance each beam.
            select_indices_array = []
            # Loop over the batch_size number of beam
            for j, b in enumerate(beam):
                b.advance(out[j, :],
                          beam_attn.data[j, :, :memory_lengths[j]])
                select_indices_array.append(
                    b.get_current_origin() + j * beam_size)
            select_indices = torch.cat(select_indices_array)

            self.model.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        # (4) Extract sentences from beam.
        for b in beam:
            scores, ks = b.sort_finished(minimum=self.n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:self.n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            results["predictions"].append(hyps)
            results["scores"].append(scores)
            results["attention"].append(attn)

        return results

    def _score_target(self, batch, memory_bank, src_lengths, data, src_map):
        tgt_in = inputters.make_features(batch, 'tgt')[:-1]

        log_probs, attn = self._decode_and_generate(
            tgt_in, memory_bank, batch, data,
            memory_lengths=src_lengths, src_map=src_map)
        tgt_pad = self.fields["tgt"].vocab.stoi[self.fields["tgt"].pad_token]

        log_probs[:, :, tgt_pad] = 0
        gold = batch.tgt[1:].unsqueeze(2)
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores

    def _report_score(self, name, score_total, words_total):
        if words_total == 0:
            msg = "%s No words predicted" % (name,)
        else:
            msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                name, score_total / words_total,
                name, math.exp(-score_total / words_total)))
        return msg

    def _report_bleu(self, tgt_path):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")
        # Rollback pointer to the beginning.
        self.out_file.seek(0)
        print()

        res = subprocess.check_output(
            "perl %s/tools/multi-bleu.perl %s" % (base_dir, tgt_path),
            stdin=self.out_file, shell=True
        ).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    def _report_rouge(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        msg = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN" % (path, tgt_path),
            shell=True, stdin=self.out_file
        ).decode("utf-8").strip()
        return msg
