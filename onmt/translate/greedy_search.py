import torch
import torch.nn.functional as F

from onmt.translate.decode_strategy import DecodeStrategy


def sample_topp(logits, keep_topp):
    sorted_logits, sorted_indices = torch.sort(logits,
                                               descending=True,
                                               dim=1)

    cumulative_probs = torch.cumsum(F.softmax(sorted_logits,
                                              dim=-1), dim=-1)
    sorted_indices_to_keep = cumulative_probs.lt(keep_topp)

    # keep indices until overflowing p
    cumsum_mask = sorted_indices_to_keep.cumsum(dim=1)
    last_included = cumsum_mask[:, -1:]
    last_included.clamp_(0, sorted_indices_to_keep.size()[1] - 1)
    sorted_indices_to_keep = sorted_indices_to_keep.scatter_(
        1, last_included, 1)

    # Set all logits that are not in the top-p to -10000.
    # This puts the probabilities close to 0.
    keep_indices = sorted_indices_to_keep.scatter(
                                1,
                                sorted_indices,
                                sorted_indices_to_keep,
                                )
    return logits.masked_fill(~keep_indices, -10000)


def sample_topk(logits, keep_topk):
    top_values, _ = torch.topk(logits, keep_topk, dim=1)
    kth_best = top_values[:, -1].view([-1, 1])
    kth_best = kth_best.repeat([1, logits.shape[1]]).float()

    # Set all logits that are not in the top-k to -10000.
    # This puts the probabilities close to 0.
    ignore = torch.lt(logits, kth_best)
    return logits.masked_fill(ignore, -10000)


def sample_with_temperature(logits, sampling_temp, keep_topk, keep_topp):
    """Select next tokens randomly from the top k possible next tokens.

    Samples from a categorical distribution over the ``keep_topk`` words using
    the category probabilities ``logits / sampling_temp``.

    Args:
        logits (FloatTensor): Shaped ``(batch_size, vocab_size)``.
            These can be logits (``(-inf, inf)``) or log-probs (``(-inf, 0]``).
            (The distribution actually uses the log-probabilities
            ``logits - logits.logsumexp(-1)``, which equals the logits if
            they are log-probabilities summing to 1.)
        sampling_temp (float): Used to scale down logits. The higher the
            value, the more likely it is that a non-max word will be
            sampled.
        keep_topk (int): This many words could potentially be chosen. The
            other logits are set to have probability 0.
        keep_topp (float): Keep most likely words until the cumulated
            probability is greater than p. If used with keep_topk: both
            conditions will be applied

    Returns:
        (LongTensor, FloatTensor):

        * topk_ids: Shaped ``(batch_size, 1)``. These are
          the sampled word indices in the output vocab.
        * topk_scores: Shaped ``(batch_size, 1)``. These
          are essentially ``(logits / sampling_temp)[topk_ids]``.
    """

    if sampling_temp == 0.0 or keep_topk == 1:
        # For temp=0.0, take the argmax to avoid divide-by-zero errors.
        # keep_topk=1 is also equivalent to argmax.
        topk_scores, topk_ids = logits.topk(1, dim=-1)
        if sampling_temp > 0:
            topk_scores /= sampling_temp
    else:
        logits = torch.div(logits, sampling_temp)
        if keep_topp > 0:
            logits = sample_topp(logits, keep_topp)
        if keep_topk > 0:
            logits = sample_topk(logits, keep_topk)
        dist = torch.distributions.Multinomial(
            logits=logits, total_count=1)
        topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)
        topk_scores = logits.gather(dim=1, index=topk_ids)
    return topk_ids, topk_scores


class GreedySearch(DecodeStrategy):
    """Select next tokens randomly from the top k possible next tokens.

    The ``scores`` attribute's lists are the score, after applying temperature,
    of the final prediction (either EOS or the final token in the event
    that ``max_length`` is reached)

    Args:
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        unk (int): See base.
        batch_size (int): See base.
        global_scorer (onmt.translate.GNMTGlobalScorer): Scorer instance.
        min_length (int): See base.
        max_length (int): See base.
        ban_unk_token (Boolean): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.
        return_attention (bool): See base.
        max_length (int): See base.
        sampling_temp (float): See
            :func:`~onmt.translate.greedy_search.sample_with_temperature()`.
        keep_topk (int): See
            :func:`~onmt.translate.greedy_search.sample_with_temperature()`.
        keep_topp (float): See
            :func:`~onmt.translate.greedy_search.sample_with_temperature()`.
        beam_size (int): Number of beams to use.
    """

    def __init__(self, pad, bos, eos, unk, batch_size, global_scorer,
                 min_length, block_ngram_repeat, exclusion_tokens,
                 return_attention, max_length, sampling_temp, keep_topk,
                 keep_topp, beam_size, ban_unk_token):
        super(GreedySearch, self).__init__(
            pad, bos, eos, unk, batch_size, beam_size, global_scorer,
            min_length, block_ngram_repeat, exclusion_tokens,
            return_attention, max_length, ban_unk_token)
        self.sampling_temp = sampling_temp
        self.keep_topk = keep_topk
        self.keep_topp = keep_topp
        self.topk_scores = None
        self.beam_size = beam_size

    def initialize(self, memory_bank, src_lengths, src_map=None, device=None,
                   target_prefix=None):
        """Initialize for decoding."""
        (fn_map_state, memory_bank,
            src_map, target_prefix) = self.initialize_tile(
                memory_bank, src_lengths, src_map, target_prefix)
        if device is None:
            device = self.get_device_from_memory_bank(memory_bank)

        super(GreedySearch, self).initialize(
            memory_bank, src_lengths, src_map, device, target_prefix)
        self.select_indices = torch.arange(
            self.batch_size*self.beam_size, dtype=torch.long, device=device)
        self.original_batch_idx = fn_map_state(torch.arange(
            self.batch_size, dtype=torch.long, device=device), dim=0)
        self.beams_scores = torch.zeros((self.batch_size*self.beam_size, 1),
                                        dtype=torch.float, device=device)
        return fn_map_state, memory_bank, self.memory_lengths, src_map

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def batch_offset(self):
        return self.select_indices

    def _pick(self, log_probs):
        """Function used to pick next tokens.

        Args:
            log_probs (FloatTensor): ``(batch_size, vocab_size)``.
        """
        # maybe fix some prediction at this step by modifying log_probs
        log_probs = self.target_prefixing(log_probs)
        topk_ids, topk_scores = sample_with_temperature(
            log_probs, self.sampling_temp, self.keep_topk, self.keep_topp)
        return topk_ids, topk_scores

    def align_select_indices(self):
        nb_finished_beams = (self.is_finished.view(-1).size(0) -
                             self.select_indices.size(0))
        if nb_finished_beams:
            self.select_indices = torch.arange(
                self.select_indices.size(0), dtype=torch.long,
                device=self.select_indices.device)

    def advance(self, log_probs, attn):
        """Select next tokens randomly from the top k possible next tokens.

        Args:
            log_probs (FloatTensor): Shaped ``(batch_size, vocab_size)``.
                These can be logits (``(-inf, inf)``) or log-probs
                (``(-inf, 0]``). (The distribution actually uses the
                log-probabilities ``logits - logits.logsumexp(-1)``,
                which equals the logits if they are log-probabilities summing
                to 1.)
            attn (FloatTensor): Shaped ``(1, B, inp_seq_len)``.
        """

        self.align_select_indices()

        self.ensure_min_length(log_probs)
        self.ensure_unk_removed(log_probs)
        self.block_ngram_repeats(log_probs)

        topk_ids, self.topk_scores = self._pick(log_probs)
        self.beams_scores += self.topk_scores

        self.is_finished = topk_ids.eq(self.eos)

        self.alive_seq = torch.cat([self.alive_seq, topk_ids], -1)
        if self.return_attention:
            if self.alive_attn is None:
                self.alive_attn = attn
            else:
                self.alive_attn = torch.cat([self.alive_attn, attn], 0)
        self.ensure_max_length()

    def update_finished(self):
        """Finalize scores and predictions."""
        # shape: (sum(~ self.is_finished), 1)
        finished_batches = self.is_finished.view(-1).nonzero(as_tuple=False)
        step = len(self)
        length_penalty = self.global_scorer.length_penalty(
            step, alpha=self.global_scorer.alpha)

        for b in finished_batches.view(-1):
            b_orig = self.original_batch_idx[b]
            score = self.beams_scores[b, 0]/length_penalty
            pred = self.alive_seq[b, 1:]
            attention = (
                self.alive_attn[:, b, :self.memory_lengths[b]]
                if self.alive_attn is not None else [])
            self.hypotheses[b_orig].append((score, pred, attention))
        self.done = self.is_finished.all()
        if self.done:
            for b in range(self.batch_size):
                best_hyp = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for score, pred, attn in best_hyp:
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)
                    self.attention[b].append(attn)
            return
        is_alive = ~self.is_finished.view(-1)
        self.alive_seq = self.alive_seq[is_alive]
        self.beams_scores = self.beams_scores[is_alive]
        if self.alive_attn is not None:
            self.alive_attn = self.alive_attn[:, is_alive]
        self.select_indices = is_alive.nonzero(as_tuple=False).view(-1)
        self.original_batch_idx = self.original_batch_idx[is_alive]
        self.maybe_update_target_prefix(self.select_indices)


class GreedySearchLM(GreedySearch):
    def update_finished(self):
        super(GreedySearchLM, self).update_finished()
        self.update_memory_lengths()

    def update_memory_lengths(self):
        is_alive = ~self.is_finished.view(-1)
        self.memory_lengths = self.memory_lengths[is_alive]

    def advance(self, log_probs, attn):
        super(GreedySearchLM, self).advance(log_probs, attn)

        # in LM task memory_lengths is associated with currently generated src
        # and therefore needs to follow the generation
        self.memory_lengths += 1

    def initialize(self, src, src_lengths, src_map=None, device=None,
                   target_prefix=None):
        """Initialize for decoding."""

        if device is None:
            device = src.device

        (fn_map_state, _, self.memory_lengths,
            src_map) = super(GreedySearchLM, self).initialize(
                None, src_lengths, src_map, device, target_prefix)
        src = fn_map_state(src, dim=1)

        return fn_map_state, src, self.memory_lengths, src_map
