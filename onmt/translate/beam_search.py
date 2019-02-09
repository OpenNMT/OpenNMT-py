import torch


class BeamSearch(object):
    """Generation beam search.

    Note that the attributes list is not exhaustive. Rather, it highlights
    tensors to document their shape. (Since the state variables' "batch"
    size decreases as beams finish, we denote this axis with a B rather than
    ``batch_size``).

    Args:
        beam_size (int): Number of beams to use.
        batch_size (int): Current batch size.
        pad (int): Magic integer in output vocab.
        bos (int): Magic integer in output vocab.
        eos (int): Magic integer in output vocab.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        mb_device (torch.device or str): Device for memory bank (encoder).
        global_scorer (onmt.translate.GNMTGlobalScorer): Scorer instance.
        min_length (int): Shortest acceptable generation, not counting
            begin-of-sentence or end-of-sentence.
        max_length (int): Longest acceptable sequence, not counting
            begin-of-sentence (presumably there has been no EOS
            yet if max_length is used as a cutoff).
        return_attention (bool): Whether to work with attention too.
        block_ngram_repeat (int): Block beams where
            ``block_ngram_repeat``-grams repeat.
        exclusion_tokens (set[str]): If a gram contains any of these
            tokens, it may repeat.
        memory_lengths (torch.LongTensor): Lengths of encodings. Used for
            masking attentions.

    Attributes:
        top_beam_finished (torch.ByteTensor): Shape ``(B,)``.
        _batch_offset (torch.LongTensor): Shape ``(B,)``.
        _beam_offset (torch.LongTensor): Shape ``(batch_size x beam_size)``.
        alive_seq (torch.LongTensor): Shape ``(B x beam_size, step)``. This
            sequence grows in the ``step`` axis on each call to
            :func:`advance()`.
        topk_log_probs (torch.FloatTensor): Shape ``(B x beam_size,)``. These
            are the scores used for the topk operation.
        alive_attn (torch.FloatTensor or NoneType): If tensor, shape is
            ``(step, B x beam_size, inp_seq_len)``, where ``inp_seq_len``
            is the (max) length of the input sequence.
        select_indices (torch.LongTensor or NoneType): Shape
            ``(B x beam_size,)``. Initialized to ``None``.
        is_finished (torch.ByteTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``.
        topk_scores (torch.FloatTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``. These are the
            scores a sequence will receive if it finishes.
        topk_ids (torch.LongTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``. These are the
            word indices of the topk predictions.
        _batch_index (torch.LongTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``.
        _prev_penalty (torch.FloatTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``.
        _coverage (torch.FloatTensor or NoneType): Shape
            ``(1, B x beam_size, inp_seq_len)``.
        hypotheses (list[list[Tuple[torch.Tensor]]]): Contains a tuple
            of score (float), sequence (long), and attention (float or None).
        scores (list[list[torch.FloatTensor]]): For each batch, holds a
            list of beam scores.
        predictions (list[list[torch.LongTensor]]): For each batch, holds a
            list of beam prediction sequences.
        attention (list[list[torch.FloatTensor or list[]]]): For each batch,
            holds a list of beam attention sequence tensors (or empty lists)
            having shape ``(step, inp_seq_len)`` where ``inp_seq_len`` is the
            length of the sample (not the max length of all inp seqs).
    """

    def __init__(self, beam_size, batch_size, pad, bos, eos, n_best, mb_device,
                 global_scorer, min_length, max_length, return_attention,
                 block_ngram_repeat, exclusion_tokens, memory_lengths,
                 stepwise_penalty):
        # magic indices
        self.pad = pad
        self.eos = eos
        self.bos = bos

        # beam parameters
        self.min_length = min_length
        self.global_scorer = global_scorer
        self.beam_size = beam_size
        self.max_length = max_length
        self.return_attention = return_attention
        self.n_best = n_best
        self.batch_size = batch_size
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens

        # result caching
        self.hypotheses = [[] for _ in range(batch_size)]
        self.predictions = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]
        self.attention = [[] for _ in range(batch_size)]

        # beam state
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        self._batch_offset = torch.arange(batch_size, dtype=torch.long)
        self._beam_offset = torch.arange(
            0, batch_size * beam_size, step=beam_size, dtype=torch.long,
            device=mb_device)
        self.alive_seq = torch.full(
            [batch_size * beam_size, 1], self.bos, dtype=torch.long,
            device=mb_device)
        self.topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (beam_size - 1), device=mb_device
        ).repeat(batch_size)
        self.alive_attn = None
        self.select_indices = None
        self.is_finished = None
        self.topk_scores = None
        self._memory_lengths = memory_lengths
        self.topk_ids = None
        self._batch_index = None
        self.done = False
        # "global state" of the old beam
        self._prev_penalty = None
        self._coverage = None

        self._stepwise_cov_pen = (
                stepwise_penalty and self.global_scorer.has_cov_pen)
        self._vanilla_cov_pen = (
            not stepwise_penalty and self.global_scorer.has_cov_pen)
        self._cov_pen = self.global_scorer.has_cov_pen

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_origin(self):
        return self.select_indices

    @property
    def current_backptr(self):
        # for testing
        return self.select_indices.view(self.batch_size, self.beam_size)\
            .fmod(self.beam_size)

    def advance(self, log_probs, attn):
        vocab_size = log_probs.size(-1)

        if self._stepwise_cov_pen and self._prev_penalty is not None:
            self.topk_log_probs += self._prev_penalty
            self.topk_log_probs -= self.global_scorer.cov_penalty(
                self._coverage + attn, self.global_scorer.beta).view(
                -1, self.beam_size)

        # force the output to be longer than self.min_length
        step = self.alive_seq.shape[1]
        if step <= self.min_length:
            log_probs[:, self.eos] = -1e20

        # Multiply probs by the beam probability.
        log_probs += self.topk_log_probs.view(-1).unsqueeze(1)

        # block ngram repeats
        if self.block_ngram_repeat > 0 and step > 1:
            # iterate over all batches, over all beams
            for bk in range(self.alive_seq.shape[0]):
                hyp = self.alive_seq[bk, 1:]
                ngrams = set()
                fail = False
                gram = []
                for i in range(step - 1):
                    # Last n tokens, n = block_ngram_repeat
                    gram = (gram + [hyp[i].item()])[-self.block_ngram_repeat:]
                    # skip the blocking if any token in gram is excluded
                    if set(gram) & self.exclusion_tokens:
                        continue
                    if tuple(gram) in ngrams:
                        fail = True
                    ngrams.add(tuple(gram))
                if fail:
                    log_probs[bk] = -10e20

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token
        length_penalty = self.global_scorer.length_penalty(
            step + 1, alpha=self.global_scorer.alpha)

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs / length_penalty
        curr_scores = curr_scores.reshape(-1, self.beam_size * vocab_size)
        self.topk_scores, self.topk_ids = curr_scores.topk(
            self.beam_size, dim=-1)

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.
        self.topk_log_probs = self.topk_scores * length_penalty

        # Resolve beam origin and true word ids.
        topk_beam_index = self.topk_ids.div(vocab_size)
        self.topk_ids = self.topk_ids.fmod(vocab_size)

        # Map beam_index to batch_index in the flat representation.
        self._batch_index = (
                topk_beam_index
                + self._beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
        self.select_indices = self._batch_index.view(-1)

        # Append last prediction.
        self.alive_seq = torch.cat(
            [self.alive_seq.index_select(0, self.select_indices),
             self.topk_ids.view(-1, 1)], -1)
        if self.return_attention or self._cov_pen:
            current_attn = attn.index_select(1, self.select_indices)
            if self.alive_attn is None:
                self.alive_attn = current_attn
                # update global state (step == 1)
                if self.global_scorer.beta > 0:  # coverage penalty
                    self._prev_penalty = torch.zeros_like(self.topk_log_probs)
                    self._coverage = current_attn
            else:
                self.alive_attn = self.alive_attn.index_select(
                    1, self.select_indices)
                self.alive_attn = torch.cat([self.alive_attn, current_attn], 0)
                # update global state (step > 1)
                self._coverage = self._coverage\
                    .index_select(1, self.select_indices) + current_attn
                self._prev_penalty = self.global_scorer.cov_penalty(
                    self._coverage, beta=self.global_scorer.beta).view(
                        -1, self.beam_size)

        if self._vanilla_cov_pen:
            # shape: (batch_size x beam_size, 1)
            cov_penalty = self.global_scorer.cov_penalty(
                self._coverage,
                beta=self.global_scorer.beta)
            self.topk_scores -= cov_penalty.view(*self.topk_scores.shape)

        self.is_finished = self.topk_ids.eq(self.eos)
        if step == self.max_length:
            self.is_finished.fill_(1)

    def update_finished(self):
        # Penalize beams that finished.
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10)
        # on real data (newstest2017) with the pretrained transformer,
        # it's faster to not move this back to the original device
        self.is_finished = self.is_finished.to('cpu')
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)
        predictions = self.alive_seq.view(
            -1, self.beam_size, self.alive_seq.size(-1))
        attention = (
            self.alive_attn.view(
                self.alive_attn.size(0), -1, self.beam_size,
                self.alive_attn.size(-1))
            if self.alive_attn is not None else None)
        non_finished_batch = []
        for i in range(self.is_finished.size(0)):
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero().view(-1)
            # Store finished hypotheses for this batch.
            for j in finished_hyp:
                self.hypotheses[b].append((
                    self.topk_scores[i, j],
                    predictions[i, j, 1:],  # Ignore start_token.
                    attention[:, i, j, :self._memory_lengths[i]]
                    if attention is not None else None))
            # End condition is the top beam finished and we can return
            # n_best hypotheses.
            if self.top_beam_finished[i] and len(
                    self.hypotheses[b]) >= self.n_best:
                best_hyp = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for n, (score, pred, attn) in enumerate(best_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)
                    self.attention[b].append(
                        attn if attn is not None else [])
            else:
                non_finished_batch.append(i)
        non_finished = torch.tensor(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return
        # Remove finished batches for the next step.
        self.top_beam_finished = self.top_beam_finished.index_select(
            0, non_finished)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        non_finished = non_finished.to(self.topk_ids.device)
        self.topk_log_probs = self.topk_log_probs.index_select(0,
                                                               non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(-1)
        self.alive_seq = predictions.index_select(0, non_finished) \
            .view(-1, self.alive_seq.size(-1))
        if self.alive_attn is not None:
            self.alive_attn = attention.index_select(1, non_finished) \
                .view(self.alive_attn.size(0),
                      -1, self.alive_attn.size(-1))
            cov = (
                self._coverage.view(
                    1, -1, self.beam_size, self._coverage.size(-1)))
            self._coverage = cov.index_select(1, non_finished) \
                .view(1, -1, self._coverage.size(-1))
            self._prev_penalty = self._prev_penalty.index_select(
                0, non_finished)
