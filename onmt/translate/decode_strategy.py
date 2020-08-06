import torch
from copy import deepcopy


class DecodeStrategy(object):
    """Base class for generation strategies.

    Args:
        pad (int): Magic integer in output vocab.
        bos (int): Magic integer in output vocab.
        eos (int): Magic integer in output vocab.
        batch_size (int): Current batch size.
        parallel_paths (int): Decoding strategies like beam search
            use parallel paths. Each batch is repeated ``parallel_paths``
            times in relevant state tensors.
        min_length (int): Shortest acceptable generation, not counting
            begin-of-sentence or end-of-sentence.
        max_length (int): Longest acceptable sequence, not counting
            begin-of-sentence (presumably there has been no EOS
            yet if max_length is used as a cutoff).
        block_ngram_repeat (int): Block beams where
            ``block_ngram_repeat``-grams repeat.
        exclusion_tokens (set[int]): If a gram contains any of these
            tokens, it may repeat.
        return_attention (bool): Whether to work with attention too. If this
            is true, it is assumed that the decoder is attentional.

    Attributes:
        pad (int): See above.
        bos (int): See above.
        eos (int): See above.
        predictions (list[list[LongTensor]]): For each batch, holds a
            list of beam prediction sequences.
        scores (list[list[FloatTensor]]): For each batch, holds a
            list of scores.
        attention (list[list[FloatTensor or list[]]]): For each
            batch, holds a list of attention sequence tensors
            (or empty lists) having shape ``(step, inp_seq_len)`` where
            ``inp_seq_len`` is the length of the sample (not the max
            length of all inp seqs).
        alive_seq (LongTensor): Shape ``(B x parallel_paths, step)``.
            This sequence grows in the ``step`` axis on each call to
            :func:`advance()`.
        is_finished (ByteTensor or NoneType): Shape
            ``(B, parallel_paths)``. Initialized to ``None``.
        alive_attn (FloatTensor or NoneType): If tensor, shape is
            ``(step, B x parallel_paths, inp_seq_len)``, where ``inp_seq_len``
            is the (max) length of the input sequence.
        target_prefix (LongTensor or NoneType): If tensor, shape is
            ``(B x parallel_paths, prefix_seq_len)``, where ``prefix_seq_len``
            is the (max) length of the pre-fixed prediction.
        min_length (int): See above.
        max_length (int): See above.
        block_ngram_repeat (int): See above.
        exclusion_tokens (set[int]): See above.
        return_attention (bool): See above.
        done (bool): See above.
    """

    def __init__(self, pad, bos, eos, batch_size, parallel_paths,
                 min_length, block_ngram_repeat, exclusion_tokens,
                 return_attention, max_length):

        # magic indices
        self.pad = pad
        self.bos = bos
        self.eos = eos

        self.batch_size = batch_size
        self.parallel_paths = parallel_paths
        # result caching
        self.predictions = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]
        self.attention = [[] for _ in range(batch_size)]

        self.alive_attn = None

        self.min_length = min_length
        self.max_length = max_length

        self.block_ngram_repeat = block_ngram_repeat
        n_paths = batch_size * parallel_paths
        self.forbidden_tokens = [dict() for _ in range(n_paths)]

        self.exclusion_tokens = exclusion_tokens
        self.return_attention = return_attention

        self.done = False

    def initialize(self, memory_bank, src_lengths, src_map=None, device=None,
                   target_prefix=None):
        """DecodeStrategy subclasses should override :func:`initialize()`.

        `initialize` should be called before all actions.
        used to prepare necessary ingredients for decode.
        """
        if device is None:
            device = torch.device('cpu')
        self.alive_seq = torch.full(
            [self.batch_size * self.parallel_paths, 1], self.bos,
            dtype=torch.long, device=device)
        self.is_finished = torch.zeros(
            [self.batch_size, self.parallel_paths],
            dtype=torch.uint8, device=device)
        if target_prefix is not None:
            seq_len, batch_size, n_feats = target_prefix.size()
            assert batch_size == self.batch_size * self.parallel_paths,\
                "forced target_prefix should've extend to same number of path!"
            target_prefix_words = target_prefix[:, :, 0].transpose(0, 1)
            target_prefix = target_prefix_words[:, 1:]  # remove bos
            # fix length constraint
            prefix_non_pad = target_prefix.ne(self.pad).sum(dim=-1).tolist()
            self.max_length += max(prefix_non_pad)
            self.min_length += min(prefix_non_pad)
        self.target_prefix = target_prefix  # NOTE: forced prefix words
        return None, memory_bank, src_lengths, src_map

    def __len__(self):
        return self.alive_seq.shape[1]

    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length:
            log_probs[:, self.eos] = -1e20

    def ensure_max_length(self):
        # add one to account for BOS. Don't account for EOS because hitting
        # this implies it hasn't been found.
        if len(self) == self.max_length + 1:
            self.is_finished.fill_(1)

    def block_ngram_repeats(self, log_probs):
        """
        We prevent the beam from going in any direction that would repeat any
        ngram of size <block_ngram_repeat> more thant once.

        The way we do it: we maintain a list of all ngrams of size
        <block_ngram_repeat> that is updated each time the beam advances, and
        manually put any token that would lead to a repeated ngram to 0.

        This improves on the previous version's complexity:
           - previous version's complexity: batch_size * beam_size * len(self)
           - current version's complexity: batch_size * beam_size

        This improves on the previous version's accuracy;
           - Previous version blocks the whole beam, whereas here we only
            block specific tokens.
           - Before the translation would fail when all beams contained
            repeated ngrams. This is sure to never happen here.
        """

        # we don't block nothing if the user doesn't want it
        if self.block_ngram_repeat <= 0:
            return

        # we can't block nothing beam's too short
        if len(self) < self.block_ngram_repeat:
            return

        n = self.block_ngram_repeat - 1
        for path_idx in range(self.alive_seq.shape[0]):
            # we check paths one by one

            current_ngram = tuple(self.alive_seq[path_idx, -n:].tolist())
            forbidden_tokens = self.forbidden_tokens[path_idx].get(
                current_ngram, None)
            if forbidden_tokens is not None:
                log_probs[path_idx, list(forbidden_tokens)] = -10e20

    def maybe_update_forbidden_tokens(self):
        """We complete and reorder the list of forbidden_tokens"""

        # we don't forbid nothing if the user doesn't want it
        if self.block_ngram_repeat <= 0:
            return

        # we can't forbid nothing if beam's too short
        if len(self) < self.block_ngram_repeat:
            return

        n = self.block_ngram_repeat

        forbidden_tokens = list()
        for path_idx, seq in zip(self.select_indices, self.alive_seq):

            # Reordering forbidden_tokens following beam selection
            # We rebuild a dict to ensure we get the value and not the pointer
            forbidden_tokens.append(
                deepcopy(self.forbidden_tokens[path_idx]))

            # Grabing the newly selected tokens and associated ngram
            current_ngram = tuple(seq[-n:].tolist())

            # skip the blocking if any token in current_ngram is excluded
            if set(current_ngram) & self.exclusion_tokens:
                continue

            forbidden_tokens[-1].setdefault(current_ngram[:-1], set())
            forbidden_tokens[-1][current_ngram[:-1]].add(current_ngram[-1])

        self.forbidden_tokens = forbidden_tokens

    def target_prefixing(self, log_probs):
        """Fix the first part of predictions with `self.target_prefix`.

        Args:
            log_probs (FloatTensor): logits of size ``(B, vocab_size)``.

        Returns:
            log_probs (FloatTensor): modified logits in ``(B, vocab_size)``.
        """
        _B, vocab_size = log_probs.size()
        step = len(self)
        if (self.target_prefix is not None and
                step <= self.target_prefix.size(1)):
            pick_idx = self.target_prefix[:, step - 1].tolist()  # (B)
            pick_coo = [[path_i, pick] for path_i, pick in enumerate(pick_idx)
                        if pick not in [self.eos, self.pad]]
            mask_pathid = [path_i for path_i, pick in enumerate(pick_idx)
                           if pick in [self.eos, self.pad]]
            if len(pick_coo) > 0:
                pick_coo = torch.tensor(pick_coo).to(self.target_prefix)
                pick_fill_value = torch.ones(
                    [pick_coo.size(0)], dtype=log_probs.dtype)
                # pickups: Tensor where specified index were set to 1, others 0
                pickups = torch.sparse_coo_tensor(
                    pick_coo.t(), pick_fill_value,
                    size=log_probs.size(), device=log_probs.device).to_dense()
                # dropdowns: opposite of pickups, 1 for those shouldn't pick
                dropdowns = torch.ones_like(pickups) - pickups
                if len(mask_pathid) > 0:
                    path_mask = torch.zeros(_B).to(self.target_prefix)
                    path_mask[mask_pathid] = 1
                    path_mask = path_mask.unsqueeze(1).to(dtype=bool)
                    dropdowns = dropdowns.masked_fill(path_mask, 0)
                # Minus dropdowns to log_probs making probabilities of
                # unspecified index close to 0
                log_probs -= 10000*dropdowns
        return log_probs

    def maybe_update_target_prefix(self, select_index):
        """We update / reorder `target_prefix` for alive path."""
        if self.target_prefix is None:
            return
        # prediction step have surpass length of given target_prefix,
        # no need to further change this attr
        if len(self) > self.target_prefix.size(1):
            return
        self.target_prefix = self.target_prefix.index_select(0, select_index)

    def advance(self, log_probs, attn):
        """DecodeStrategy subclasses should override :func:`advance()`.

        Advance is used to update ``self.alive_seq``, ``self.is_finished``,
        and, when appropriate, ``self.alive_attn``.
        """

        raise NotImplementedError()

    def update_finished(self):
        """DecodeStrategy subclasses should override :func:`update_finished()`.

        ``update_finished`` is used to update ``self.predictions``,
        ``self.scores``, and other "output" attributes.
        """

        raise NotImplementedError()
