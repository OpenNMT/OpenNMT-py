import torch
from onmt.translate import penalties
from onmt.translate.decode_strategy import DecodeStrategy
import warnings


class BeamSearchBase(DecodeStrategy):
    """Generation beam search.

    Note that the attributes list is not exhaustive. Rather, it highlights
    tensors to document their shape. (Since the state variables' "batch"
    size decreases as beams finish, we denote this axis with a B rather than
    ``batch_size``).

    Args:
        beam_size (int): Number of beams to use (see base ``parallel_paths``).
        batch_size (int): See base.
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        unk (int): See base.
        start (int): See base.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        global_scorer (onmt.translate.GNMTGlobalScorer): Scorer instance.
        min_length (int): See base.
        max_length (int): See base.
        return_attention (bool): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.

    Attributes:
        _batch_offset (LongTensor): Shape ``(B,)``.
        _beam_offset (LongTensor): Shape ``(batch_size x beam_size,)``.
        alive_seq (LongTensor): See base.
        topk_log_probs (FloatTensor): Shape ``(B, beam_size,)``. These
            are the scores used for the topk operation.
        src_len (LongTensor): Lengths of encodings. Used for
            masking attentions.
        select_indices (LongTensor or NoneType): Shape
            ``(B x beam_size,)``. This is just a flat view of the
            ``_batch_index``.
        topk_scores (FloatTensor): Shape
            ``(B, beam_size)``. These are the
            scores a sequence will receive if it finishes.
        topk_ids (LongTensor): Shape ``(B, beam_size)``. These are the
            word indices of the topk predictions.
        _batch_index (LongTensor): Shape ``(B, beam_size)``.
        _prev_penalty (FloatTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``.
        _coverage (FloatTensor or NoneType): Shape
            ``(1, B x beam_size, inp_seq_len)``.
        hypotheses (list[list[Tuple[Tensor]]]): Contains a tuple
            of score (float), sequence (long), and attention (float or None).
    """

    def __init__(
        self,
        beam_size,
        batch_size,
        pad,
        bos,
        eos,
        unk,
        start,
        n_best,
        global_scorer,
        min_length,
        max_length,
        return_attention,
        block_ngram_repeat,
        exclusion_tokens,
        stepwise_penalty,
        ratio,
        ban_unk_token,
    ):
        super(BeamSearchBase, self).__init__(
            pad,
            bos,
            eos,
            unk,
            start,
            batch_size,
            beam_size,
            global_scorer,
            min_length,
            block_ngram_repeat,
            exclusion_tokens,
            return_attention,
            max_length,
            ban_unk_token,
        )
        # beam parameters
        self.beam_size = beam_size
        self.n_best = n_best
        self.ratio = ratio

        # beam state
        self._batch_offset = torch.arange(batch_size, dtype=torch.long)

        self.select_indices = None
        self.done = False
        # "global state" of the old beam
        self._prev_penalty = None
        self._coverage = None

        self._stepwise_cov_pen = stepwise_penalty and self.global_scorer.has_cov_pen
        self._vanilla_cov_pen = not stepwise_penalty and self.global_scorer.has_cov_pen
        self._cov_pen = self.global_scorer.has_cov_pen

        self.src_len = None

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def initialize_(self, enc_out, src_map, device, target_prefix):
        super(BeamSearchBase, self).initialize(device, target_prefix)
        self.best_scores = [-1e10 for _ in range(self.batch_size)]
        self._beam_offset = torch.arange(
            0,
            self.batch_size * self.beam_size,
            step=self.beam_size,
            dtype=torch.long,
            device=device,
        )
        self.topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (self.beam_size - 1), device=device)
            .repeat(self.batch_size)
            .reshape(self.batch_size, self.beam_size)
        )
        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty(
            (self.batch_size, self.beam_size), dtype=torch.float, device=device
        )
        """
        self.topk_ids = torch.empty(
            (self.batch_size, self.beam_size), dtype=torch.long, device=device
        )

        self._batch_index = torch.empty(
            [self.batch_size, self.beam_size], dtype=torch.long, device=device
        )
        """

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_backptr(self):
        # for testing
        return self.select_indices.view(self.batch_size, self.beam_size).fmod(
            self.beam_size
        )

    @property
    def batch_offset(self):
        return self._batch_offset

    def _pick(self, log_probs):
        """Take a token pick decision for a step.

        Args:
            log_probs (FloatTensor): (B * beam_size, vocab_size)

        Returns:
            topk_scores (FloatTensor): (B, beam_size)
            topk_ids (LongTensor): (B, beam_size)
        """
        vocab_size = log_probs.size(-1)
        # maybe fix some prediction at this step by modifying log_probs
        log_probs = self.target_prefixing(log_probs)

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs.reshape(-1, self.beam_size * vocab_size)

        topk_scores, topk_ids = torch.topk(curr_scores, self.beam_size, dim=-1)
        # after this we get topk_ids between 0 and beam_size*vocab_size
        # topk_ids // vocab_size => indice in beam
        # topk_ids % vocab_size => true vocab indice

        return topk_scores, topk_ids

    def beams_non_finished(self, i, topk_scores_list, predictions, attention, step):
        # using lists instead of tensors for topk_scores and is_finished make things faster
        if any(self.is_finished_list[i]):
            b = self._batch_offset[i]
            # Store finished hypotheses for this example in the batch.
            for j in [
                k for k, fin in enumerate(self.is_finished_list[i]) if fin
            ]:  # Beam level: finished beam j in example i of batch
                if self.ratio > 0:
                    s = topk_scores_list[i][j] / (step + 1)
                    self.best_scores[b] = max(s, self.best_scores[b])
                self.hypotheses[b].append(
                    (
                        topk_scores_list[i][j],
                        predictions[i, j, 1:],  # Ignore start_token.
                        attention[i, j, :, : self.src_len[i]]
                        if attention is not None
                        else None,
                    )
                )

            # End condition is the top beam finished and we can return
            # n_best hypotheses.
            if self.ratio > 0:
                pred_len = self.src_len[i] * self.ratio
                finish_flag = (
                    (topk_scores_list[i][0] / pred_len) <= self.best_scores[b]
                ) or all(self.is_finished_list[i])
            else:
                # early stop when top beam is finished
                finish_flag = self.is_finished_list[i][0]

            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                self.hypotheses[b] = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True
                )
                for score, pred, attn in self.hypotheses[b][: self.n_best]:
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)  # ``(batch, n_best,)``
                    self.attention[b].append(attn if attn is not None else [])
                return False
            else:
                return True
        else:
            return True

    def update_finished(self):
        # Penalize beams that finished.
        _B_old = self.topk_log_probs.shape[0]
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance
        # this is required to pursue finished beams in non finished batches
        self.topk_log_probs.masked_fill_(
            torch.tensor(self.is_finished_list, device=self.topk_log_probs.device),
            -65504,
        )
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)
        attention = (
            self.alive_attn.view(
                _B_old, self.beam_size, step - 1, self.alive_attn.size(-1)
            )
            if self.alive_attn is not None
            else None
        )

        topk_scores_list = self.topk_scores.tolist()
        non_finished_batch = [
            i
            for i in range(len(self.is_finished_list))
            if self.beams_non_finished(
                i, topk_scores_list, predictions, attention, step
            )
        ]

        non_finished = torch.tensor(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]
        self.remove_finished_batches(
            _B_new, _B_old, non_finished, predictions, attention, step
        )

        # reset the selection for the next step
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.src_len = self.src_len[self.select_indices]
        self.maybe_update_target_prefix(self.select_indices)

    def remove_finished_batches(
        self, _B_new, _B_old, non_finished, predictions, attention, step
    ):
        # Remove finished batches for the next step.
        self._batch_offset = self._batch_offset[non_finished]  # CPU
        non_finished = non_finished.to(self.topk_log_probs.device)
        self.topk_log_probs = self.topk_log_probs[non_finished]
        self._batch_index = self._batch_index[non_finished]
        self.alive_seq = predictions[non_finished].view(-1, self.alive_seq.size(-1))

        if self.alive_attn is not None:
            inp_seq_len = self.alive_attn.size(-1)
            self.alive_attn = attention[non_finished].view(
                _B_new * self.beam_size, step - 1, inp_seq_len
            )
            if self._cov_pen:
                self._coverage = self._coverage.view(
                    _B_old, self.beam_size, 1, inp_seq_len
                )[non_finished].view(_B_new * self.beam_size, 1, inp_seq_len)
                if self._stepwise_cov_pen:
                    self._prev_penalty = self._prev_penalty[non_finished]

    def advance(self, log_probs, attn):
        vocab_size = log_probs.size(-1)

        # using integer division to get an integer _B without casting
        _B = log_probs.shape[0] // self.beam_size

        if self._stepwise_cov_pen and self._prev_penalty is not None:
            self.topk_log_probs += self._prev_penalty
            self.topk_log_probs -= self.global_scorer.cov_penalty(
                self._coverage + attn, self.global_scorer.beta
            ).view(_B, self.beam_size)

        # force the output to be longer than self.min_length
        step = len(self)
        self.ensure_min_length(log_probs)
        self.ensure_unk_removed(log_probs)
        # Multiply probs by the beam probability.
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token
        length_penalty = self.global_scorer.length_penalty(
            step + 1, alpha=self.global_scorer.alpha
        )
        if length_penalty != 1:
            curr_scores = log_probs / length_penalty
        else:
            curr_scores = log_probs

        # Avoid any direction that would repeat unwanted ngrams
        self.block_ngram_repeats(curr_scores)

        # Pick up candidate token by curr_scores
        self.topk_scores, self.topk_ids = self._pick(curr_scores)

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.
        self.topk_log_probs = self.topk_scores * length_penalty

        # Resolve beam origin and map to batch index flat representation.
        self._batch_index = self.topk_ids // vocab_size + self._beam_offset[
            :_B
        ].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)
        self.topk_ids %= vocab_size

        # Append last prediction to reordered alive sequence
        self.alive_seq = torch.cat(
            [
                self.alive_seq[self.select_indices],
                self.topk_ids.view(_B * self.beam_size, 1),
            ],
            -1,
        )

        self.maybe_update_forbidden_tokens()

        if self.return_attention or self._cov_pen:
            current_attn = attn[self.select_indices]
            if step == 1:
                self.alive_attn = current_attn
                # update global state (step == 1)
                if self._cov_pen:  # coverage penalty
                    self._prev_penalty = torch.zeros_like(self.topk_log_probs)
                    self._coverage = current_attn
            else:
                self.alive_attn = self.alive_attn[self.select_indices]
                self.alive_attn = torch.cat([self.alive_attn, current_attn], 1)
                # update global state (step > 1)
                if self._cov_pen:
                    self._coverage = self._coverage[self.select_indices]
                    self._coverage += current_attn
                    self._prev_penalty = self.global_scorer.cov_penalty(
                        self._coverage, beta=self.global_scorer.beta
                    ).view(_B, self.beam_size)

        if self._vanilla_cov_pen:
            # shape: (batch_size x beam_size, 1)
            cov_penalty = self.global_scorer.cov_penalty(
                self._coverage, beta=self.global_scorer.beta
            )
            self.topk_scores -= cov_penalty.view(_B, self.beam_size).float()

        self.is_finished_list = self.topk_ids.eq(self.eos).tolist()

        self.ensure_max_length()


class BeamSearch(BeamSearchBase):
    """
    Beam search for seq2seq/encoder-decoder models
    """

    def initialize(
        self, enc_out, src_len, src_map=None, device=None, target_prefix=None
    ):
        """Initialize for decoding.
        Repeat src objects `beam_size` times.
        """

        (fn_map_state, enc_out, src_map, target_prefix) = self.initialize_tile(
            enc_out, src_len, src_map, target_prefix
        )
        if device is None:
            device = self.get_device_from_enc_out(enc_out)

        super(BeamSearch, self).initialize_(enc_out, src_map, device, target_prefix)

        return fn_map_state, enc_out, src_map


class BeamSearchLM(BeamSearchBase):
    """
    Beam search for language/decoder only models
    """

    def initialize(self, src, src_len, src_map=None, device=None, target_prefix=None):
        """Initialize for decoding.
        Repeat src objects `beam_size` times.
        """
        (fn_map_state, _, src_map, target_prefix) = self.initialize_tile(
            None, src_len, src_map, target_prefix
        )
        if device is None:
            device = src.device

        super(BeamSearchLM, self).initialize_(
            None,
            src_map=src_map,
            device=device,
            target_prefix=target_prefix,
        )

        return fn_map_state, src, src_map

    def advance(self, log_probs, attn):
        super(BeamSearchLM, self).advance(log_probs, attn)

        # in LM task src_len is associated with currently generated src
        # and therefore needs to follow the generation
        self.src_len += 1


class GNMTGlobalScorer(object):
    """NMT re-ranking.

    Args:
       alpha (float): Length parameter.
       beta (float):  Coverage parameter.
       length_penalty (str): Length penalty strategy.
       coverage_penalty (str): Coverage penalty strategy.

    Attributes:
        alpha (float): See above.
        beta (float): See above.
        length_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        coverage_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        has_cov_pen (bool): See :class:`penalties.PenaltyBuilder`.
        has_len_pen (bool): See :class:`penalties.PenaltyBuilder`.
    """

    @classmethod
    def from_opt(cls, opt):
        return cls(opt.alpha, opt.beta, opt.length_penalty, opt.coverage_penalty)

    def __init__(self, alpha, beta, length_penalty, coverage_penalty):
        self._validate(alpha, beta, length_penalty, coverage_penalty)
        self.alpha = alpha
        self.beta = beta
        penalty_builder = penalties.PenaltyBuilder(coverage_penalty, length_penalty)
        self.has_cov_pen = penalty_builder.has_cov_pen
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty

        self.has_len_pen = penalty_builder.has_len_pen
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty

    @classmethod
    def _validate(cls, alpha, beta, length_penalty, coverage_penalty):
        # these warnings indicate that either the alpha/beta
        # forces a penalty to be a no-op, or a penalty is a no-op but
        # the alpha/beta would suggest otherwise.
        if length_penalty is not None and alpha == 0.0:
            warnings.warn(
                "Using length penalty with alpha==0 "
                "is equivalent to using length penalty none."
            )
        if coverage_penalty is None or coverage_penalty == "none":
            if beta != 0:
                warnings.warn(
                    "Non-default `beta` with no coverage penalty. "
                    "`beta` has no effect."
                )
        else:
            # using some coverage penalty
            if beta == 0.0:
                warnings.warn(
                    "Non-default coverage penalty with beta==0 "
                    "is equivalent to using coverage penalty none."
                )
