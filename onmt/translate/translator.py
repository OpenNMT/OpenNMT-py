#!/usr/bin/env python
""" Translator Class and builder """
import torch
from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import pad_sequence
import codecs
from time import time
from math import exp
from itertools import count, zip_longest
from copy import deepcopy
import onmt.model_builder
import onmt.decoders.ensemble
from onmt.constants import DefaultTokens
from onmt.translate.beam_search import BeamSearch, BeamSearchLM
from onmt.translate.greedy_search import GreedySearch, GreedySearchLM
from onmt.utils.misc import tile, set_random_seed, report_matrix
from onmt.utils.alignment import extract_alignment, build_align_pharaoh
from onmt.modules.copy_generator import collapse_copy_scores
from onmt.constants import ModelTask
from onmt.transforms import TransformPipe


def build_translator(opt, device_id=0, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, "w+", "utf-8")

    load_test_model = (
        onmt.decoders.ensemble.load_test_model
        if len(opt.models) > 1
        else onmt.model_builder.load_test_model
    )

    vocabs, model, model_opt = load_test_model(opt, device_id)

    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)

    if model_opt.model_task == ModelTask.LANGUAGE_MODEL:
        translator = GeneratorLM.from_opt(
            model,
            vocabs,
            opt,
            model_opt,
            global_scorer=scorer,
            out_file=out_file,
            report_align=opt.report_align,
            report_score=report_score,
            logger=logger,
        )
    else:
        translator = Translator.from_opt(
            model,
            vocabs,
            opt,
            model_opt,
            global_scorer=scorer,
            out_file=out_file,
            report_align=opt.report_align,
            report_score=report_score,
            logger=logger,
        )
    return translator


class Inference(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (onmt.modules.NMTModel): NMT model to use for translation
        vocabs (dict[str, Vocab]): A dict
            mapping each side's Vocab.
        gpu (int): GPU device. Set to negative for no GPU.
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        random_sampling_temp (float): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        dump_beam (bool): Debugging option.
        block_ngram_repeat (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        ignore_when_blocking (set or frozenset): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        replace_unk (bool): Replace unknown token.
        tgt_file_prefix (bool): Force the predictions begin with provided -tgt.
        data_type (str): Source data type.
        verbose (bool): Print/log every translation.
        report_time (bool): Print/log total time/frequency.
        copy_attn (bool): Use copy attention.
        global_scorer (onmt.translate.GNMTGlobalScorer): Translation
            scoring/reranking object.
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        report_score (bool) : Whether to report scores
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
        self,
        model,
        vocabs,
        gpu=-1,
        n_best=1,
        min_length=0,
        max_length=100,
        max_length_ratio=1.5,
        ratio=0.0,
        beam_size=30,
        random_sampling_topk=0,
        random_sampling_topp=0.0,
        random_sampling_temp=1.0,
        stepwise_penalty=None,
        dump_beam=False,
        block_ngram_repeat=0,
        ignore_when_blocking=frozenset(),
        replace_unk=False,
        ban_unk_token=False,
        tgt_file_prefix=False,
        phrase_table="",
        data_type="text",
        verbose=False,
        report_time=False,
        copy_attn=False,
        global_scorer=None,
        out_file=None,
        report_align=False,
        gold_align=False,
        report_score=True,
        logger=None,
        seed=-1,
        with_score=False,
        return_gold_log_probs=False,
    ):
        self.model = model
        self.vocabs = vocabs
        self._tgt_vocab = vocabs["tgt"]
        self._tgt_eos_idx = vocabs["tgt"].lookup_token(DefaultTokens.EOS)
        self._tgt_pad_idx = vocabs["tgt"].lookup_token(DefaultTokens.PAD)
        self._tgt_bos_idx = vocabs["tgt"].lookup_token(DefaultTokens.BOS)
        self._tgt_unk_idx = vocabs["tgt"].lookup_token(DefaultTokens.UNK)
        self._tgt_sep_idx = vocabs["tgt"].lookup_token(DefaultTokens.SEP)
        self._tgt_start_with = vocabs["tgt"].lookup_token(vocabs["decoder_start_token"])
        self._tgt_vocab_len = len(self._tgt_vocab)

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = (
            torch.device("cuda", self._gpu) if self._use_cuda else torch.device("cpu")
        )

        self.n_best = n_best
        self.max_length = max_length
        self.max_length_ratio = max_length_ratio

        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk
        self.sample_from_topp = random_sampling_topp

        self.min_length = min_length
        self.ban_unk_token = ban_unk_token
        self.ratio = ratio
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {self._tgt_vocab[t] for t in self.ignore_when_blocking}
        self.replace_unk = replace_unk
        if self.replace_unk and not self.model.decoder.attentional:
            raise ValueError("replace_unk requires an attentional decoder.")
        self.tgt_file_prefix = tgt_file_prefix
        self.phrase_table = phrase_table
        self.data_type = data_type
        self.verbose = verbose
        self.report_time = report_time

        self.copy_attn = copy_attn

        self.global_scorer = global_scorer
        if self.global_scorer.has_cov_pen and not self.model.decoder.attentional:
            raise ValueError("Coverage penalty requires an attentional decoder.")
        self.out_file = out_file
        self.report_align = report_align
        self.gold_align = gold_align
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False
        self._filter_pred = None

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": [],
            }

        set_random_seed(seed, self._use_cuda)
        self.with_score = with_score

        self.return_gold_log_probs = return_gold_log_probs

    @classmethod
    def from_opt(
        cls,
        model,
        vocabs,
        opt,
        model_opt,
        global_scorer=None,
        out_file=None,
        report_align=False,
        report_score=True,
        logger=None,
    ):
        """Alternate constructor.

        Args:
            model (onmt.modules.NMTModel): See :func:`__init__()`.
            vocabs (dict[str, Vocab]): See
                :func:`__init__()`.
            opt (argparse.Namespace): Command line options
            model_opt (argparse.Namespace): Command line options saved with
                the model checkpoint.
            global_scorer (onmt.translate.GNMTGlobalScorer): See
                :func:`__init__()`..
            out_file (TextIO or codecs.StreamReaderWriter): See
                :func:`__init__()`.
            report_align (bool) : See :func:`__init__()`.
            report_score (bool) : See :func:`__init__()`.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """
        # TODO: maybe add dynamic part
        cls.validate_task(model_opt.model_task)

        return cls(
            model,
            vocabs,
            gpu=opt.gpu,
            n_best=opt.n_best,
            min_length=opt.min_length,
            max_length=opt.max_length,
            max_length_ratio=opt.max_length_ratio,
            ratio=opt.ratio,
            beam_size=opt.beam_size,
            random_sampling_topk=opt.random_sampling_topk,
            random_sampling_topp=opt.random_sampling_topp,
            random_sampling_temp=opt.random_sampling_temp,
            stepwise_penalty=opt.stepwise_penalty,
            dump_beam=opt.dump_beam,
            block_ngram_repeat=opt.block_ngram_repeat,
            ignore_when_blocking=set(opt.ignore_when_blocking),
            replace_unk=opt.replace_unk,
            ban_unk_token=opt.ban_unk_token,
            tgt_file_prefix=opt.tgt_file_prefix,
            phrase_table=opt.phrase_table,
            data_type=opt.data_type,
            verbose=opt.verbose,
            report_time=opt.report_time,
            copy_attn=model_opt.copy_attn,
            global_scorer=global_scorer,
            out_file=out_file,
            report_align=report_align,
            gold_align=opt.gold_align,
            report_score=report_score,
            logger=logger,
            seed=opt.seed,
            with_score=opt.with_score,
        )

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _gold_score(
        self, batch, enc_out, src_len, use_src_map, enc_final_hs, batch_size, src
    ):
        if "tgt" in batch.keys() and not self.tgt_file_prefix:
            gs, glp = self._score_target(
                batch, enc_out, src_len, batch["src_map"] if use_src_map else None
            )
            self.model.decoder.init_state(src, enc_out, enc_final_hs)
        else:
            gs = [0] * batch_size
            glp = None
        return gs, glp

    def _translate(
        self,
        infer_iter,
        transform=None,
        attn_debug=False,
        align_debug=False,
        phrase_table="",
    ):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Args:
            infer_iter: tensored batch iterator from DynamicDatasetIter
            attn_debug (bool): enables the attention logging
            align_debug (bool): enables the word alignment logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        transform_pipe = (
            TransformPipe.build_from([transform[name] for name in transform])
            if transform
            else None
        )
        xlation_builder = onmt.translate.TranslationBuilder(
            self.vocabs,
            self.n_best,
            self.replace_unk,
            self.phrase_table,
        )

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        start_time = time()

        def _maybe_retranslate(translations, batch):
            """Here we handle the cases of mismatch in number of segments
            between source and target. We re-translate seg by seg."""
            inds, perm = torch.sort(batch["ind_in_bucket"])
            trans_copy = deepcopy(translations)
            inserted_so_far = 0
            for j, trans in enumerate(translations):
                if (trans.src == self._tgt_sep_idx).sum().item() != trans.pred_sents[
                    0
                ].count(DefaultTokens.SEP):
                    self._log("Mismatch in number of ((newline))")
                    # those two should be the same except feat dim
                    # batch['src'][perm[j], :, :])
                    # trans.src

                    # we rebuild a small batch made of the sub-segments
                    # in the long segment.
                    idx = (trans.src == self._tgt_sep_idx).nonzero()
                    sub_src = []
                    start_idx = 0
                    for i in range(len(idx)):
                        end_idx = idx[i]
                        sub_src.append(batch["src"][perm[j], start_idx:end_idx, :])
                        start_idx = end_idx + 1
                    end_idx = (
                        batch["src"][perm[j], :, 0].ne(self._tgt_pad_idx).sum() - 1
                    )
                    sub_src.append(batch["src"][perm[j], start_idx:end_idx, :])
                    t_sub_src = pad_sequence(
                        sub_src, batch_first=True, padding_value=self._tgt_pad_idx
                    )
                    t_sub_src_len = t_sub_src[:, :, 0].ne(self._tgt_pad_idx).sum(1)
                    t_sub_src_ind = torch.tensor(
                        [i for i in range(len(sub_src))], dtype=torch.int16
                    )
                    device = batch["src"].device
                    t_sub_batch = {
                        "src": t_sub_src.to(device),
                        "srclen": t_sub_src_len.to(device),
                        "ind_in_bucket": t_sub_src_ind.to(device),
                    }
                    # new sub-batch ready to be translated
                    sub_data = self.translate_batch(t_sub_batch, attn_debug)
                    sub_trans = xlation_builder.from_batch(sub_data)

                    # we re-insert the sub-batch in the initial translations
                    trans_copy[j + inserted_so_far] = sub_trans[0]
                    for i in range(1, len(sub_src)):
                        trans_copy.insert(j + i + inserted_so_far, sub_trans[i])
                    inserted_so_far += len(sub_src) - 1
            return trans_copy

        def _process_bucket(bucket_translations):
            bucket_scores = []
            bucket_predictions = []
            bucket_score = 0
            bucket_words = 0
            bucket_gold_score = 0
            bucket_gold_words = 0
            voc_src = self.vocabs["src"].ids_to_tokens
            bucket_translations = sorted(
                bucket_translations, key=lambda x: x.ind_in_bucket
            )
            for trans in bucket_translations:
                bucket_scores += [trans.pred_scores[: self.n_best]]
                bucket_score += trans.pred_scores[0]
                bucket_words += len(trans.pred_sents[0])
                if "tgt" in batch.keys():
                    bucket_gold_score += trans.gold_score
                    bucket_gold_words += len(trans.gold_sent) + 1

                n_best_preds = [
                    " ".join(pred) for pred in trans.pred_sents[: self.n_best]
                ]

                if self.report_align:
                    align_pharaohs = [
                        build_align_pharaoh(align)
                        for align in trans.word_aligns[: self.n_best]
                    ]
                    n_best_preds_align = [
                        " ".join(align[0]) for align in align_pharaohs
                    ]
                    n_best_preds = [
                        pred + DefaultTokens.ALIGNMENT_SEPARATOR + align
                        for pred, align in zip(n_best_preds, n_best_preds_align)
                    ]

                if transform_pipe is not None:
                    n_best_preds = transform_pipe.batch_apply_reverse(n_best_preds)

                bucket_predictions += [n_best_preds]

                if self.with_score:
                    n_best_scores = [
                        score.item() for score in trans.pred_scores[: self.n_best]
                    ]
                    out_all = [
                        pred + "\t" + str(score)
                        for (pred, score) in zip(n_best_preds, n_best_scores)
                    ]
                    self.out_file.write("\n".join(out_all) + "\n")
                else:
                    self.out_file.write("\n".join(n_best_preds) + "\n")
                self.out_file.flush()

                if self.verbose:
                    srcs = [voc_src[tok] for tok in trans.src[: trans.srclen]]
                    sent_number = next(counter)
                    output = trans.log(sent_number, src_raw=srcs)
                    self._log(output)

                if attn_debug:
                    preds = trans.pred_sents[0]
                    preds.append(DefaultTokens.EOS)
                    attns = trans.attns[0].tolist()
                    if self.data_type == "text":
                        srcs = [
                            voc_src[tok] for tok in trans.src[: trans.srclen].tolist()
                        ]
                    else:
                        srcs = [str(item) for item in range(len(attns[0]))]
                    output = report_matrix(srcs, preds, attns)
                    self._log(output)

                if align_debug:
                    if self.gold_align:
                        tgts = trans.gold_sent
                    else:
                        tgts = trans.pred_sents[0]
                    align = trans.word_aligns[0].tolist()
                    if self.data_type == "text":
                        srcs = [
                            voc_src[tok] for tok in trans.src[: trans.srclen].tolist()
                        ]
                    else:
                        srcs = [str(item) for item in range(len(align[0]))]
                    output = report_matrix(srcs, tgts, align)
                    self._log(output)

            return (
                bucket_scores,
                bucket_predictions,
                bucket_score,
                bucket_words,
                bucket_gold_score,
                bucket_gold_words,
            )

        bucket_translations = []
        prev_idx = 0

        for batch, bucket_idx in infer_iter:

            batch_data = self.translate_batch(batch, attn_debug)

            translations = xlation_builder.from_batch(batch_data)
            if (
                not isinstance(self, GeneratorLM)
                and self._tgt_sep_idx != self._tgt_unk_idx
                and (batch["src"] == self._tgt_sep_idx).any().item()
            ):
                # For seq2seq when we need to force doc to spit the same number of sents
                translations = _maybe_retranslate(translations, batch)

            bucket_translations += translations

            if (
                not isinstance(infer_iter, list)
                and len(bucket_translations) >= infer_iter.bucket_size
            ):
                bucket_idx += 1

            if bucket_idx != prev_idx:
                prev_idx = bucket_idx
                (
                    bucket_scores,
                    bucket_predictions,
                    bucket_score,
                    bucket_words,
                    bucket_gold_score,
                    bucket_gold_words,
                ) = _process_bucket(bucket_translations)
                all_scores += bucket_scores
                all_predictions += bucket_predictions
                pred_score_total += bucket_score
                pred_words_total += bucket_words
                gold_score_total += bucket_gold_score
                gold_words_total += bucket_gold_words
                bucket_translations = []

        if len(bucket_translations) > 0:
            (
                bucket_scores,
                bucket_predictions,
                bucket_score,
                bucket_words,
                bucket_gold_score,
                bucket_gold_words,
            ) = _process_bucket(bucket_translations)
            all_scores += bucket_scores
            all_predictions += bucket_predictions
            pred_score_total += bucket_score
            pred_words_total += bucket_words
            gold_score_total += bucket_gold_score
            gold_words_total += bucket_gold_words

        end_time = time()

        if self.report_score:
            msg = self._report_score("PRED", pred_score_total, len(all_scores))
            self._log(msg)
            if "tgt" in batch.keys() and not self.tgt_file_prefix:
                msg = self._report_score("GOLD", gold_score_total, len(all_scores))
                self._log(msg)

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total translation time (s): %.1f" % total_time)
            self._log(
                "Average translation time (ms): %.1f"
                % (total_time / len(all_predictions) * 1000)
            )
            self._log("Tokens per second: %.1f" % (pred_words_total / total_time))

        if self.dump_beam:
            import json

            json.dump(
                self.translator.beam_accum,
                codecs.open(self.dump_beam, "w", "utf-8"),
            )

        return all_scores, all_predictions

    def _score(self, infer_iter):
        self.with_scores = True
        score_res = []
        processed_bucket = {}
        prev_bucket_idx = 0
        for batch, bucket_idx in infer_iter:
            if bucket_idx != prev_bucket_idx:
                prev_bucket_idx += 1
                score_res += [item for _, item in sorted(processed_bucket.items())]
                processed_bucket = {}
            batch_data = self.translate_batch(batch, attn_debug=False, scoring=True)
            batch_gold_scores = batch_data["gold_score"].cpu().numpy().tolist()
            batch_tgt_lengths = batch["tgtlen"].cpu().numpy().tolist()
            batch_inds_in_bucket = batch["ind_in_bucket"]
            if self.return_gold_log_probs:
                batch_gold_log_probs = (
                    batch_data["gold_log_probs"].cpu().numpy().tolist()
                )
            else:
                batch_gold_log_probs = [
                    None for i, _ in enumerate(batch_inds_in_bucket)
                ]
            for i, ind in enumerate(batch_inds_in_bucket):
                processed_bucket[ind] = [
                    batch_gold_scores[i],
                    batch_gold_log_probs[i],
                    batch_tgt_lengths[i],
                ]
        if processed_bucket:
            score_res += [item for _, item in sorted(processed_bucket.items())]
        return score_res

    def _align_pad_prediction(self, predictions, bos, pad):
        """
        Padding predictions in batch and add BOS.

        Args:
            predictions (List[List[Tensor]]): `(batch, n_best,)`, for each src
                sequence contain n_best tgt predictions all of which ended with
                eos id.
            bos (int): bos index to be used.
            pad (int): pad index to be used.

        Return:
            batched_nbest_predict (torch.LongTensor): `(batch, n_best, tgt_l)`
        """
        dtype, device = predictions[0][0].dtype, predictions[0][0].device
        flatten_tgt = [best.tolist() for bests in predictions for best in bests]
        paded_tgt = torch.tensor(
            list(zip_longest(*flatten_tgt, fillvalue=pad)),
            dtype=dtype,
            device=device,
        ).T
        bos_tensor = torch.full([paded_tgt.size(0), 1], bos, dtype=dtype, device=device)
        full_tgt = torch.cat((bos_tensor, paded_tgt), dim=-1)
        batched_nbest_predict = full_tgt.view(
            len(predictions), -1, full_tgt.size(-1)
        )  # (batch, n_best, tgt_l)
        return batched_nbest_predict

    def _report_score(self, name, score_total, nb_sentences):
        # In the case of length_penalty = none we report the total logprobs
        # divided by the number of sentence to get an approximation of the
        # per sentence logprob. We also return the corresponding ppl
        # When a length_penalty is used eg: "avg" or "wu" since logprobs
        # are normalized per token we report the per line per token logprob
        # and the corresponding "per word perplexity"
        if nb_sentences == 0:
            msg = "%s No translations" % (name,)
        else:
            score = score_total / nb_sentences
            try:
                ppl = exp(-score_total / nb_sentences)
            except OverflowError:
                ppl = float("inf")
            msg = "%s SCORE: %.4f, %s PPL: %.2f NB SENTENCES: %d" % (
                name,
                score,
                name,
                ppl,
                nb_sentences,
            )
        return msg

    def _decode_and_generate(
        self,
        decoder_in,
        enc_out,
        batch,
        src_len,
        src_map=None,
        step=None,
        batch_offset=None,
        return_attn=False,
    ):
        if self.copy_attn:
            # Turn any copied words into UNKs.
            decoder_in = decoder_in.masked_fill(
                decoder_in.gt(self._tgt_vocab_len - 1), self._tgt_unk_idx
            )

        # Decoder forward, takes [batch, tgt_len, nfeats] as input
        # and [batch, src_len, hidden] as enc_out
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch

        dec_out, dec_attn = self.model.decoder(
            decoder_in,
            enc_out,
            src_len=src_len,
            step=step,
            return_attn=self.global_scorer.has_cov_pen or return_attn,
        )
        # Generator forward.
        if not self.copy_attn:
            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None
            scores = self.model.generator(dec_out.squeeze(1))
            log_probs = log_softmax(scores, dim=-1)  # we keep float16 if FP16
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [batch_size, tgt_len, vocab ] when full sentence
        else:
            attn = dec_attn["copy"]
            scores = self.model.generator(
                dec_out.view(-1, dec_out.size(2)),
                attn.view(-1, attn.size(2)),
                src_map,
            )
            # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
            if batch_offset is None:
                scores = scores.view(-1, len(batch["srclen"]), scores.size(-1))
                scores = scores.transpose(0, 1).contiguous()
            else:
                scores = scores.view(-1, self.beam_size, scores.size(-1))
            # at this point scores is batch first (dim=0)
            scores = collapse_copy_scores(
                scores,
                batch,
                self._tgt_vocab,
                batch_dim=0,
            )
            scores = scores.view(-1, decoder_in.size(1), scores.size(-1))
            log_probs = scores.squeeze(1).log()
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [batch_size, tgt_len, vocab ] when full sentence
        return log_probs, attn

    def translate_batch(self, batch, attn_debug):
        """Translate a batch of sentences."""
        raise NotImplementedError

    def _score_target(self, batch, enc_out, src_len, src_map):
        raise NotImplementedError

    def report_results(
        self,
        gold_score,
        gold_log_probs,
        batch,
        batch_size,
        decode_strategy,
    ):
        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": batch,
            "gold_score": gold_score,
            "gold_log_probs": gold_log_probs,
        }

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        if self.report_align:
            results["alignment"] = self._align_forward(
                batch, decode_strategy.predictions
            )
        else:
            results["alignment"] = [[] for _ in range(batch_size)]
        return results


class Translator(Inference):
    @classmethod
    def validate_task(cls, task):
        if task != ModelTask.SEQ2SEQ:
            raise ValueError(
                f"Translator does not support task {task}."
                f" Tasks supported: {ModelTask.SEQ2SEQ}"
            )

    def _align_forward(self, batch, predictions):
        """
        For a batch of input and its prediction, return a list of batch predict
        alignment src indice Tensor in size ``(batch, n_best,)``.
        """

        # (0) add BOS and padding to tgt prediction
        if "tgt" in batch.keys() and self.gold_align:
            self._log("Computing alignments with gold target")
            batch_tgt_idxs = batch["tgt"].transpose(1, 2)
        else:
            batch_tgt_idxs = self._align_pad_prediction(
                predictions, bos=self._tgt_bos_idx, pad=self._tgt_pad_idx
            )
        tgt_mask = (
            batch_tgt_idxs.eq(self._tgt_pad_idx)
            | batch_tgt_idxs.eq(self._tgt_eos_idx)
            | batch_tgt_idxs.eq(self._tgt_bos_idx)
        )

        n_best = batch_tgt_idxs.size(1)
        # (1) Encoder forward.
        src, enc_states, enc_out, src_len = self._run_encoder(batch)

        # (2) Repeat src objects `n_best` times.
        # We use batch_size x n_best, get ``(batch * n_best, src_len, nfeat)``
        src = tile(src, n_best, dim=0)
        if enc_states is not None:
            # Quick fix. Transformers return None as enc_states.
            # enc_states are only used later on to init decoder's state
            # but are never used in Transformer decoder, so we can skip
            enc_states = tile(enc_states, n_best, dim=0)
        if isinstance(enc_out, tuple):
            enc_out = tuple(tile(x, n_best, dim=0) for x in enc_out)
        else:
            enc_out = tile(enc_out, n_best, dim=0)
        src_len = tile(src_len, n_best)  # ``(batch * n_best,)``

        # (3) Init decoder with n_best src,
        self.model.decoder.init_state(src, enc_out, enc_states)
        # reshape tgt to ``(len, batch * n_best, nfeat)``
        # it should be done in a better way
        tgt = batch_tgt_idxs.view(-1, batch_tgt_idxs.size(-1)).T.unsqueeze(-1)
        dec_in = tgt[:-1].transpose(0, 1)  # exclude last target from inputs
        # here dec_in is batch first
        _, attns = self.model.decoder(dec_in, enc_out, src_len=src_len, with_align=True)

        alignment_attn = attns["align"]  # ``(B, tgt_len-1, src_len)``
        # masked_select
        align_tgt_mask = tgt_mask.view(-1, tgt_mask.size(-1))
        prediction_mask = align_tgt_mask[:, 1:]  # exclude bos to match pred
        # get aligned src id for each prediction's valid tgt tokens
        alignement = extract_alignment(alignment_attn, prediction_mask, src_len, n_best)
        return alignement

    def translate_batch(self, batch, attn_debug):
        """Translate a batch of sentences."""
        if self.max_length_ratio > 0:
            max_length = int(
                min(self.max_length, batch["src"].size(1) * self.max_length_ratio + 5)
            )
        else:
            max_length = self.max_length
        with torch.no_grad():
            if self.sample_from_topk != 0 or self.sample_from_topp != 0:
                decode_strategy = GreedySearch(
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    start=self._tgt_start_with,
                    n_best=self.n_best,
                    batch_size=len(batch["srclen"]),
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    return_attention=attn_debug or self.replace_unk,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    keep_topp=self.sample_from_topp,
                    beam_size=self.beam_size,
                    ban_unk_token=self.ban_unk_token,
                )
            else:
                # TODO: support these blacklisted features
                assert not self.dump_beam
                decode_strategy = BeamSearch(
                    self.beam_size,
                    batch_size=len(batch["srclen"]),
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    start=self._tgt_start_with,
                    n_best=self.n_best,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=max_length,
                    return_attention=attn_debug or self.replace_unk,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=self.ratio,
                    ban_unk_token=self.ban_unk_token,
                )
            return self._translate_batch_with_strategy(batch, decode_strategy)

    def _run_encoder(self, batch):
        src = batch["src"]
        src_len = batch["srclen"]
        batch_size = len(batch["srclen"])

        enc_out, enc_final_hs, src_len = self.model.encoder(src, src_len)

        if src_len is None:
            assert not isinstance(
                enc_out, tuple
            ), "Ensemble decoding only supported for text data"
            src_len = (
                torch.Tensor(batch_size).type_as(enc_out).long().fill_(enc_out.size(1))
            )
        return src, enc_final_hs, enc_out, src_len

    def _translate_batch_with_strategy(self, batch, decode_strategy):
        """Translate a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.

        Returns:
            results (dict): The translation results.
        """
        # (0) Prep the components of the search.
        use_src_map = self.copy_attn
        parallel_paths = decode_strategy.parallel_paths  # beam_size

        batch_size = len(batch["srclen"])

        # (1) Run the encoder on the src.
        src, enc_final_hs, enc_out, src_len = self._run_encoder(batch)

        self.model.decoder.init_state(src, enc_out, enc_final_hs)

        gold_score, gold_log_probs = self._gold_score(
            batch,
            enc_out,
            src_len,
            use_src_map,
            enc_final_hs,
            batch_size,
            src,
        )

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = batch["src_map"] if use_src_map else None
        target_prefix = batch["tgt"] if self.tgt_file_prefix else None
        (fn_map_state, enc_out, src_map) = decode_strategy.initialize(
            enc_out, src_len, src_map, target_prefix=target_prefix
        )

        if fn_map_state is not None:
            self.model.decoder.map_state(fn_map_state)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(-1, 1, 1)

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                enc_out,
                batch,
                src_len=decode_strategy.src_len,
                src_map=src_map,
                step=step,
                batch_offset=decode_strategy.batch_offset,
                return_attn=decode_strategy.return_attention,
            )

            decode_strategy.advance(log_probs, attn)
            any_finished = any(
                [any(sublist) for sublist in decode_strategy.is_finished_list]
            )
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(enc_out, tuple):
                    enc_out = tuple(x[select_indices] for x in enc_out)
                else:
                    enc_out = enc_out[select_indices]

                if src_map is not None:
                    src_map = src_map[select_indices]

            if parallel_paths > 1 or any_finished:
                self.model.decoder.map_state(lambda state, dim: state[select_indices])

        return self.report_results(
            gold_score,
            gold_log_probs,
            batch,
            batch_size,
            decode_strategy,
        )

    def _score_target(self, batch, enc_out, src_len, src_map):
        tgt = batch["tgt"]
        tgt_in = tgt[:, :-1, :]

        log_probs, attn = self._decode_and_generate(
            tgt_in,
            enc_out,
            batch,
            src_len=src_len,
            src_map=src_map,
        )

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt[:, 1:, :]
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=1).view(-1)
        return gold_scores, None


class GeneratorLM(Inference):
    @classmethod
    def validate_task(cls, task):
        if task != ModelTask.LANGUAGE_MODEL:
            raise ValueError(
                f"GeneratorLM does not support task {task}."
                f" Tasks supported: {ModelTask.LANGUAGE_MODEL}"
            )

    def _align_forward(self, batch, predictions):
        """
        For a batch of input and its prediction, return a list of batch predict
        alignment src indice Tensor in size ``(batch, n_best,)``.
        """
        raise NotImplementedError

    def translate_batch(self, batch, attn_debug, scoring=False):
        """Translate a batch of sentences."""
        max_length = 0 if scoring else self.max_length
        with torch.no_grad():
            if self.sample_from_topk != 0 or self.sample_from_topp != 0:
                decode_strategy = GreedySearchLM(
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    start=self._tgt_start_with,
                    n_best=self.n_best,
                    batch_size=len(batch["srclen"]),
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    return_attention=attn_debug or self.replace_unk,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    keep_topp=self.sample_from_topp,
                    beam_size=self.beam_size,
                    ban_unk_token=self.ban_unk_token,
                )
            else:
                # TODO: support these blacklisted features
                assert not self.dump_beam
                decode_strategy = BeamSearchLM(
                    self.beam_size,
                    batch_size=len(batch["srclen"]),
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    start=self._tgt_start_with,
                    n_best=self.n_best,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=max_length,
                    return_attention=attn_debug or self.replace_unk,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=self.ratio,
                    ban_unk_token=self.ban_unk_token,
                )
            return self._translate_batch_with_strategy(batch, decode_strategy)

    @classmethod
    def split_src_to_prevent_padding(cls, src, src_len):
        min_len_batch = torch.min(src_len).item()
        target_prefix = None
        if min_len_batch > 0 and min_len_batch < src.size(1):
            target_prefix = src[:, min_len_batch:, :]
            src = src[:, :min_len_batch, :]
            src_len[:] = min_len_batch
        return src, src_len, target_prefix

    def tile_to_beam_size_after_initial_step(self, fn_map_state, log_probs):
        if fn_map_state is not None:
            log_probs = fn_map_state(log_probs, dim=0)
            self.model.decoder.map_state(fn_map_state)
            log_probs = log_probs[:, -1, :]
        return log_probs

    def _translate_batch_with_strategy(self, batch, decode_strategy, left_pad=True):
        """Translate a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.

        Returns:
            results (dict): The translation results.
        """
        # (0) Prep the components of the search.
        use_src_map = self.copy_attn
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        batch_size = len(batch["srclen"])

        # (1) split src into src and target_prefix to avoid padding.
        src = batch["src"]
        src_len = batch["srclen"]

        if left_pad:
            target_prefix = None
        else:
            src, src_len, target_prefix = self.split_src_to_prevent_padding(
                src, src_len
            )

        # (2) init decoder
        self.model.decoder.init_state(src, None, None)
        gold_score, gold_log_probs = self._gold_score(
            batch, None, src_len, use_src_map, None, batch_size, src
        )

        # (3) prep decode_strategy. Possibly repeat src objects.
        src_map = batch["src_map"] if use_src_map else None
        (fn_map_state, src, src_map) = decode_strategy.initialize(
            src,
            src_len,
            src_map,
            target_prefix=target_prefix,
        )

        # (4) Begin decoding step by step:
        # beg_time = time()
        for step in range(decode_strategy.max_length):
            decoder_input = (
                src if step == 0 else decode_strategy.current_predictions.view(-1, 1, 1)
            )
            log_probs, attn = self._decode_and_generate(
                decoder_input,
                None,
                batch,
                src_len=decode_strategy.src_len,
                src_map=src_map,
                step=step if step == 0 else step + max(src_len.tolist()),
                batch_offset=decode_strategy.batch_offset,
            )

            if step == 0:
                log_probs = self.tile_to_beam_size_after_initial_step(
                    fn_map_state, log_probs
                )

            decode_strategy.advance(log_probs, attn)
            any_finished = any(
                [any(sublist) for sublist in decode_strategy.is_finished_list]
            )
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break
            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if src_map is not None:
                    src_map = src_map[select_indices]

            if parallel_paths > 1 or any_finished:
                # select indexes in model state/cache
                self.model.decoder.map_state(lambda state, dim: state[select_indices])
            # if step == 0:
            #    print("step0 time: ", time() - beg_time)

        return self.report_results(
            gold_score,
            gold_log_probs,
            batch,
            batch_size,
            decode_strategy,
        )

    def _score_target(self, batch, enc_out, src_len, src_map):
        src = batch["src"]
        src_len = batch["srclen"]
        tgt = batch["tgt"]

        log_probs, attn = self._decode_and_generate(
            src,
            None,
            batch,
            src_len=src_len,
            src_map=src_map,
        )

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold_log_probs = log_probs.gather(2, tgt)
        gold_scores = gold_log_probs.sum(dim=1).view(-1)

        if self.return_gold_log_probs:
            return gold_scores, gold_log_probs

        return gold_scores, None
