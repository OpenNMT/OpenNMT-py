import torch

from onmt.translate.decode_strategy import DecodeStrategy


def sample_with_temperature(logits, sampling_temp, keep_topk):
    """Select next tokens randomly from the top k possible next tokens.

    Samples from a categorical distribution over the ``keep_topk`` words using
    the category probabilities ``logits / sampling_temp``.

    Args:
        logits (torch.FloatTensor): Shaped ``(batch_size, vocab_size)``.
            These can be logits (``(-inf, inf)``) or log-probs (``(-inf, 0]``).
            (The distribution actually uses the log-probabilities
            ``logits - logits.logsumexp(-1)``, which equals the logits if
            they are log-probabilities summing to 1.)
        sampling_temp (float): Used to scale down logits. The higher the
            value, the more likely it is that a non-max word will be
            sampled.
        keep_topk (int): This many words could potentially be chosen. The
            other logits are set to have probability 0.

    Returns:
        topk_ids (torch.LongTensor): Shaped ``(batch_size, 1)``. These are
            the sampled word indices in the output vocab.
        topk_scores (torch.FloatTensor): Shaped ``(batch_size, 1)``. These
            are essentially ``(logits / sampling_temp)[topk_ids]``.
    """

    if sampling_temp == 0.0 or keep_topk == 1:
        # For temp=0.0, take the argmax to avoid divide-by-zero errors.
        # keep_topk=1 is also equivalent to argmax.
        topk_scores, topk_ids = logits.topk(1, dim=-1)
    else:
        logits = torch.div(logits, sampling_temp)

        if keep_topk > 0:
            top_values, top_indices = torch.topk(logits, keep_topk, dim=1)
            kth_best = top_values[:, -1].view([-1, 1])
            kth_best = kth_best.repeat([1, logits.shape[1]]).float()

            # Set all logits that are not in the top-k to -10000.
            # This puts the probabilities close to 0.
            ignore = torch.lt(logits, kth_best)
            logits = logits.masked_fill(ignore, -10000)

        dist = torch.distributions.Multinomial(
            logits=logits, total_count=1)
        topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)
        topk_scores = logits.gather(dim=1, index=topk_ids)
    return topk_ids, topk_scores


class RandomSampling(DecodeStrategy):
    """Select next tokens randomly from the top k possible next tokens.

    Args:
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        batch_size (int): See base.
        device (torch.device or str): See base ``device``.
        min_length (int): See base.
        max_length (int): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.
        return_attention (bool): See base.
        max_length (int): See base.
        sampling_temp (float): See :func:`sample_with_temperature()`.
        keep_topk (int): See :func:`sample_with_temperature()`.
        memory_length (torch.LongTensor): Lengths of encodings. Used for
            masking attention.
    """

    # NOTE: Currently this class doesn't return "final" scores or any form
    # of Pr(EOS|pred). That is to say, the scores returned by RandomSampling
    # # are just the scores of the last token (in a batched setting, that
    # isn't even necessarily EOS since no early stopping is implemented).

    def __init__(self, pad, bos, eos, batch_size, device,
                 min_length, block_ngram_repeat, exclusion_tokens,
                 return_attention, max_length, sampling_temp, keep_topk,
                 memory_length):
        super(RandomSampling, self).__init__(
            pad, bos, eos, batch_size, device, 1,
            min_length, block_ngram_repeat, exclusion_tokens,
            return_attention, max_length)
        self.sampling_temp = sampling_temp
        self.keep_topk = keep_topk
        self.topk_scores = None
        self.memory_length = memory_length
        self.batch_size = batch_size
        self.select_indices = torch.arange(self.batch_size,
                                           dtype=torch.long, device=device)

    def advance(self, log_probs, attn):
        """Select next tokens randomly from the top k possible next tokens.

        Args:
            log_probs (torch.FloatTensor): Shaped ``(batch_size, vocab_size)``.
                These can be logits (``(-inf, inf)``) or log-probs
                (``(-inf, 0]``). (The distribution actually uses the
                log-probabilities ``logits - logits.logsumexp(-1)``,
                which equals the logits if they are log-probabilities summing
                to 1.)
            attn (torch.FloatTensor): Shaped ``(1, B, inp_seq_len)``.
        """

        self.ensure_min_length(log_probs)
        topk_ids, self.topk_scores = sample_with_temperature(
            log_probs, self.sampling_temp, self.keep_topk)

        self.alive_seq = torch.cat([self.alive_seq, topk_ids], -1)
        if self.return_attention:
            if self.alive_attn is None:
                self.alive_attn = attn
            else:
                self.alive_attn = torch.cat([self.alive_attn, attn], 0)
        self.ensure_max_length()

    def update_finished(self):
        """Finalize scores and predictions."""
        assert self.is_finished.all()
        for b in range(self.batch_size):
            self.scores[b].append(self.topk_scores[b, 0])
            self.predictions[b].append(self.alive_seq[b, 1:])
            self.attention[b].append(
                self.alive_attn[:, b, :self.memory_length[b]]
                if self.alive_attn is not None else [])
