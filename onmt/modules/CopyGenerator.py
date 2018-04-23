import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda

import onmt
import onmt.io
from onmt.Utils import aeq


class CopyGenerator(nn.Module):
    """Generator module that additionally considers copying
    words directly from the source.

    The main idea is that we have an extended "dynamic dictionary".
    It contains `|tgt_dict|` words plus an arbitrary number of
    additional words introduced by the source sentence.
    For each source sentence we have a `src_map` that maps
    each source word to an index in `tgt_dict` if it known, or
    else to an extra word.

    The copy generator is an extended version of the standard
    generator that computse three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of instead copying a
      word from the source, computed using a bernoulli
    * :math:`p_{copy}` the probility of copying a word instead.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       tgt_dict (Vocab): output target dictionary

    """

    def __init__(self, input_size, tgt_dict):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, len(tgt_dict))
        self.linear_copy = nn.Linear(input_size, 1)
        self.tgt_dict = tgt_dict

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by compying
        source words.

        Args:
           hidden (`FloatTensor`): hidden outputs `[batch*tlen, input_size]`
           attn (`FloatTensor`): attn for each `[batch*tlen, input_size]`
           src_map (`FloatTensor`):
             A sparse indicator matrix mapping each source word to
             its index in the "extended" vocab containing.
             `[src_len, batch, extra_words]`
        """
        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.tgt_dict.stoi[onmt.io.PAD_WORD]] = -float('inf')
        prob = F.softmax(logits)

        # Probability of copying p(z=1) batch.
        p_copy = F.sigmoid(self.linear_copy(hidden))
        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob,  1 - p_copy.expand_as(prob))
        mul_attn = torch.mul(attn, p_copy.expand_as(attn))
        copy_prob = torch.bmm(mul_attn.view(-1, batch, slen)
                              .transpose(0, 1),
                              src_map.transpose(0, 1)).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)


class CopyGeneratorCriterion(object):
    def __init__(self, vocab_size, force_copy, pad, eps=1e-20):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = vocab_size
        self.pad = pad

    def __call__(self, scores, align, target):
        # Compute unks in align and target for readability
        align_unk = align.eq(0).float()
        align_not_unk = align.ne(0).float()
        target_unk = target.eq(0).float()
        target_not_unk = target.ne(0).float()

        # Copy probability of tokens in source
        out = scores.gather(1, align.view(-1, 1) + self.offset).view(-1)
        # Set scores for unk to 0 and add eps
        out = out.mul(align_not_unk) + self.eps
        # Get scores for tokens in target
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)

        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            # Add score for non-unks in target
            out = out + tmp.mul(target_not_unk)
            # Add score for when word is unk in both align and tgt
            out = out + tmp.mul(align_unk).mul(target_unk)
        else:
            # Forced copy. Add only probability for not-copied tokens
            out = out + tmp.mul(align_unk)

        # Drop padding.
        loss = -out.log().mul(target.ne(self.pad).float())
        return loss


class CopyGeneratorLossCompute(onmt.Loss.LossComputeBase):
    """
    Copy Generator Loss Computation.
    """

    def __init__(self, generator, tgt_vocab,
                 force_copy, normalize_by_length,
                 eps=1e-20):
        super(CopyGeneratorLossCompute, self).__init__(
            generator, tgt_vocab)

        # We lazily load datasets when there are more than one, so postpone
        # the setting of cur_dataset.
        self.cur_dataset = None
        self.force_copy = force_copy
        self.normalize_by_length = normalize_by_length
        self.criterion = CopyGeneratorCriterion(len(tgt_vocab), force_copy,
                                                self.padding_idx)

    def _make_shard_state(self, batch, output, range_, attns):
        """ See base class for args description. """
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "copy_attn": attns.get("copy"),
            "align": batch.alignment[range_[0] + 1: range_[1]]
        }

    def _compute_loss(self, batch, output, target, copy_attn, align):
        """
        Compute the loss. The args must match self._make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(self._bottle(output),
                                self._bottle(copy_attn),
                                batch.src_map)
        loss = self.criterion(scores, align, target)
        scores_data = scores.data.clone()
        scores_data = onmt.io.TextDataset.collapse_copy_scores(
            self._unbottle(scores_data, batch.batch_size),
            batch, self.tgt_vocab, self.cur_dataset.src_vocabs)
        scores_data = self._bottle(scores_data)

        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.data.clone()
        correct_mask = target_data.eq(0) * align.data.ne(0)
        correct_copy = (align.data + len(self.tgt_vocab)) * correct_mask.long()
        target_data = target_data + correct_copy

        # Compute sum of perplexities for stats
        loss_data = loss.sum().data.clone()
        stats = self._stats(loss_data, scores_data, target_data)

        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            # Compute Sequence Lengths
            pad_ix = batch.dataset.fields['tgt'].vocab.stoi[onmt.io.PAD_WORD]
            tgt_lens = batch.tgt.ne(pad_ix).float().sum(0)
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss, stats


class PointerGenerator(CopyGenerator):
    """PointerGenerator as described in Paulus et al., (2017)
       It is similar to the `CopyGenerator` with the difference that it
       shares weights with the target embedding matrix.
    """

    def __init__(self, input_size, tgt_vocab, embeddings):
        super(PointerGenerator, self).__init__(input_size, tgt_vocab)
        self.input_size = input_size
        self.embeddings = embeddings
        W_emb = embeddings.weight
        self.linear_copy = nn.Linear(self.input_size, 1)

        n_emb, emb_dim = list(W_emb.size())

        # (2.4) Sharing decoder weights
        self.emb_proj = nn.Linear(emb_dim, self.input_size, bias=False)
        self.b_out = nn.Parameter(torch.Tensor(n_emb, 1))
        self.tanh = nn.Tanh()
        self._W_out = None

        # refresh W_out matrix after each backward pass
        self.register_backward_hook(self.refresh_W_out)

    def refresh_W_out(self, *args, **kwargs):
        self.W_out(True)

    def W_out(self, refresh=False):
        """ Sect. (2.4) Sharing decoder weights
            The function returns the W_out matrix which is a projection of the
            target embedding weight matrix.
            The W_out matrix needs to recalculated after each backward pass,
            which is done automatically. This is done to avoid calculating it
            at each decoding step (which usually leads to OOM)

            Returns:
                W_out (FloaTensor): [n_emb, 3*dim]
        """
        if self._W_out is None or refresh:
            _ = self.emb_proj(self.embeddings.weight)
            self._W_out = self.tanh(_)
        return self._W_out

    def linear(self, V):
        """Calculate the output projection of `v` as in eq. (9)
            Args:
                V (FloatTensor): [bs, 3*dim]
            Returns:
                logits (FloatTensor): logits = W_out * V + b_out, [3*dim]
        """
        W = self.W_out()
        logits = (W.matmul(V.t()) + self.b_out).t()
        return logits


class EachStepGeneratorLossCompute(CopyGeneratorLossCompute):
    def __init__(self, generator, tgt_vocab, force_copy, eps=1e-20):
        super(EachStepGeneratorLossCompute, self).__init__(
            generator, tgt_vocab, force_copy, eps)
        self.tgt_vocab = tgt_vocab

    def remove_oov(self, pred):
        """Remove out-of-vocabulary tokens
           usefull when we wants to use predictions (that contains oov due
           to copy mechanisms) as next input.
           i.e. pred[i] == 0 foreach i such as pred[i] > tgt_vocab_size
        """
        return pred.masked_fill_(pred.gt(len(self.tgt_vocab) - 1), 0)

    def compute_loss(self, batch, output, target, copy_attn, align, src,
                     prediction_type="greedy"):
        """
            align:      [bs]
            target:     [bs]
            copy_attn:  [bs x src_len]
            output:     [bs x 3*dim]
        """
        align = align.view(-1)
        target = target.view(-1)
        # GENERATOR: generating scores
        # scores: [bs x vocab + c_vocab]
        scores = self.generator(
            output,
            copy_attn,
            batch.src_map)

        # FAST COPY SCORES COLLAPSE
        # We collapse scores using only tensor operations for performance
        # This is critical since it will be executed at each decoding steps
        _src_map = batch.src_map.float().data.cuda()
        _scores = scores.data.clone()

        _src = src.clone().data
        offset = len(self.tgt_vocab)
        src_l, bs, c_vocab = _src_map.size()

        # [bs x src_len], mask of src_idx being in tgt_vocab
        src_invoc_mask = (_src.lt(offset) * _src.gt(1)).float()

        # [bs x c_voc], mask of cvocab_idx related to invoc src token
        cvoc_invoc_mask = src_invoc_mask.unsqueeze(1) \
                                        .bmm(_src_map.transpose(0, 1)) \
                                        .squeeze(1) \
                                        .gt(0)

        # [bs x src_len], copy scores of invoc src tokens
        # [bs x 1 x cvocab] @bmm [bs x cvocab x src_len] = [bs x 1 x src_len]
        src_copy_scores = _scores[:, offset:].unsqueeze(1) \
                                             .bmm(_src_map.transpose(0, 1)
                                                          .transpose(1, 2)) \
                                             .squeeze()

        # [bs x src_len], invoc src tokens, or 1 (=pad)
        # NOTE: we assume that 1 is the pad token
        src_token_invoc = _src.clone().masked_fill_(1-src_invoc_mask.byte(), 1)

        src_token_invoc = src_token_invoc.view(bs, -1)
        src_copy_scores = src_copy_scores.view(bs, -1)

        _scores.scatter_add_(
            1, src_token_invoc.long(), src_copy_scores)

        _scores[:, offset:] *= (1 - cvoc_invoc_mask.float())
        _scores[:, 1] = 0

        _collapsed_scores = _scores
        collapsed_scores = _collapsed_scores

        # CRITERION & PREDICTION: Predicting & Calculating the loss
        if prediction_type == "greedy":
            _, pred = collapsed_scores.max(1)
            pred = torch.autograd.Variable(pred)
            loss = self.criterion(scores, align, target).sum()
            loss_data = loss.data.clone()

        elif prediction_type == "sample":
            d = torch.distributions.Categorical(
                collapsed_scores[:, :len(self.tgt_vocab)])
            # TODO check if this hack is mandatory
            # in this context target=1 if continue generation, 0 else:
            # kinda hacky but seems to work
            pred = torch.autograd.Variable(d.sample()) * target

            # NOTE we use collapsed scores that account copy
            loss = self.criterion(scores, align, pred)
            loss_data = loss.sum().data
        else:
            raise ValueError("Incorrect prediction_type %s" % prediction_type)

        if output.is_cuda():
            pred.cuda()

        # FIXING TARGET TO TAKE COPY INTO ACCOUNT
        correct_target = target.data.clone()
        correct_mask = correct_target.eq(0) * align.data.ne(0)
        correct_copy = (align.data + len(self.tgt_vocab)) * correct_mask.long()
        correct_target = correct_target + correct_copy

        stats = self._stats(loss_data, collapsed_scores, correct_target)
        return loss, pred, stats
