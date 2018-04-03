"""
Implementation of "A Deep Reinforced Model for Abstractive Summarization"
Romain Paulus, Caiming Xiong, Richard Rocheri
https://arxiv.org/abs/1705.04304
"""

import torch
import torch.nn as nn
import onmt
import onmt.modules
import onmt.Models
import onmt.Trainer
import onmt.Loss
from onmt.modules import CopyGeneratorLossCompute, CopyGenerator


class RougeScorer:
    def __init__(self):
        import rouge as R
        self.rouge = R.Rouge(stats=["f"], metrics=[
                             "rouge-1", "rouge-2", "rouge-l"])

    def _score(self, hyps, refs):
        scores = self.rouge.get_scores(hyps, refs)
        # NOTE: here we use score = r1 * r2 * rl
        #       I'm not sure how relevant it is
        single_scores = [_['rouge-1']['f']*_['rouge-2']['f']*_['rouge-l']['f']
                         for _ in scores]
        return single_scores

    def score(self, sample_pred, greedy_pred, tgt):
        """
            sample_pred: LongTensor [bs x len]
            greedy_pred: LongTensor [bs x len]
            tgt: LongTensor [bs x len]
        """
        def tens2sen(t):
            sentences = []
            for s in t:
                sentence = []
                for wt in s:
                    word = wt.data[0]
                    if word in [0, 3]:
                        break
                    sentence += [str(word)]
                if len(sentence) == 0:
                    # NOTE just a trick not to score empty sentence
                    #      this has not consequence
                    sentence = ["0", "0", "0"]
                sentences += [" ".join(sentence)]
            return sentences

        s_hyps = tens2sen(sample_pred)
        g_hyps = tens2sen(greedy_pred)
        refs = tens2sen(tgt)
        sample_scores = self._score(s_hyps, refs)
        greedy_scores = self._score(g_hyps, refs)

        ts = torch.Tensor(sample_scores)
        gs = torch.Tensor(greedy_scores)

        return (gs - ts)


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
        nonan(scores, "compute_loss.scores")

        # Experimental:
        # fast copy collapse:
        # Dataset.collapse_copy_scores is very usefull in order
        # to sum copy scores for tokens that are in vocabulary
        # but using dataset.collapse_copy_scores at each step is
        # inefficient.
        # We do the same only using tensor operations
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
        src_token_invoc = _src.clone().masked_fill_(1-src_invoc_mask.byte(), 1)

        src_token_invoc = src_token_invoc.view(bs, -1)
        src_copy_scores = src_copy_scores.view(bs, -1)

        _scores.scatter_add_(
            1, src_token_invoc.long(), src_copy_scores)

        _scores[:, offset:] *= (1-cvoc_invoc_mask.float())
        _scores[:, 1] = 0

        _scores_data = _scores
        scores_data = _scores_data
        #
        # CRITERION & PREDICTION: Predicting & Calculating the loss
        #
        if prediction_type == "greedy":
            _, pred = scores_data.max(1)
            pred = torch.autograd.Variable(pred)
            loss = self.criterion(scores, align, target).sum()
            loss_data = loss.data.clone()

        elif prediction_type == "sample":
            d = torch.distributions.Categorical(
                scores_data[:, :len(self.tgt_vocab)])
            # in this context target=1 if continue generation, 0 else:
            # kinda hacky but seems to work
            pred = torch.autograd.Variable(d.sample()) * target

            # NOTE we use collapsed scores that account copy
            loss = self.criterion(scores, align, pred)
            loss_data = loss.sum().data
        else:
            raise ValueError("Incorrect prediction_type %s" % prediction_type)
        nonan(loss, "loss")
        nonan(pred, "pred")
        pred.cuda()

        #
        # FIXING TARGET
        #
        target_data = target.data.clone()
        correct_mask = target_data.eq(0) * align.data.ne(0)
        correct_copy = (align.data + len(self.tgt_vocab)) * correct_mask.long()
        target_data = target_data + correct_copy

        stats = self._stats(loss_data, scores_data, target_data)
        return loss, pred, stats


class RTrainer(onmt.Trainer.Trainer):
    """Special Trainer for the Reinforced Model
       It uses an older version of the Trainer
       The main difference is that we do not compute loss because the decoder
       is expected to do it.

    """

    def __init__(self, model, train_loss,
                 valid_loss, optim, trunc_size):
        self.model = model

        # TODO Remove this
        # self.train_iter = train_iter
        # self.valid_iter = valid_iter
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter, epoch, report_func=None):
        # NOTE quick workaround
        self.train_iter = train_iter
        total_stats = onmt.Statistics()
        report_stats = onmt.Statistics()

        for i, batch in enumerate(self.train_iter):
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            trunc_size = self.trunc_size if self.trunc_size else target_size

            dec_state = None
            _, src_lengths = batch.src

            src = onmt.io.make_features(batch, 'src')
            tgt_outer = onmt.io.make_features(batch, 'tgt')
            report_stats.n_src_words += src_lengths.sum()
            alignment = batch.alignment

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]
                batch.alignment = alignment[j + 1: j + trunc_size]

                # 2. & 3. F-prop and compute loss
                self.model.zero_grad()
                loss, batch_stats, dec_state = self.model(src, tgt,
                                                          src_lengths,
                                                          batch,
                                                          self.train_loss,
                                                          dec_state)
                loss.backward()
                # 4. Update the parameters and statistics.
                self.optim.step()

                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

            if report_func is not None:
                report_func(epoch, i, len(self.train_iter),
                            total_stats.start_time, self.optim.lr,
                            report_stats)
                report_stats = onmt.Statistics()

        return total_stats

    def validate(self, valid_iter):
        """ Called for each epoch to validate. """
        # NOTE quick workaround
        self.valid_iter = valid_iter

        # Set model in validating mode.
        self.model.eval()

        stats = onmt.Statistics()

        for batch in self.valid_iter:
            _, src_lengths = batch.src
            src = onmt.io.make_features(batch, 'src')
            tgt = onmt.io.make_features(batch, 'tgt')

            batch.alignment = batch.alignment[1:]
            # F-prop through the model.
            _, batch_stats, _ = self.model(
                src, tgt, src_lengths, batch, self.valid_loss)
            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats


def nonan(variable, name):
    d = variable.data
    nan = (d != d)
    if not nan.sum() == 0:
        print("NaN values in %s: %s" % (name, str(d)))
        inan = nan.max(0)
        print("Occuring at index: ", inan)

        i = inan[1][0]
        print("First occurence (with previous/next 5 values): ", d[i-5:i+5, :])
        raise ValueError()


def nparams(_):
    return sum([p.nelement() for p in _.parameters()])


class _Module(nn.Module):
    def __init__(self, opt):
        super(_Module, self).__init__()
        self.opt = opt

    def maybe_cuda(self, o):
        """o may be a Variable or a Tensor
        """
        if len(self.opt.gpuid) >= 1:
            return o.cuda()
        return o

    def mkvar(self, tensor, requires_grad=False):
        return self.maybe_cuda(
            torch.autograd.Variable(tensor, requires_grad=requires_grad))


def assert_size(v, size_list):
    """Check that variable(s) have size() == size_list
       v may be a variable, a tensor or a list
    """
    if type(v) not in [tuple, list]:
        v = [v]

    for variable in v:
        _friendly_aeq(real=list(variable.size()), expected=size_list)


def _friendly_aeq(real, expected):
    assert real == expected, "got %s expected: %s" % (str(real), str(expected))


class IntraAttention(_Module):
    """IntraAttention Module as in sect. (2)
    """

    def __init__(self, opt, dim, temporal=False):
        super(IntraAttention, self).__init__(opt)
        self.dim = dim
        self.temporal = temporal
        self.linear = nn.Linear(dim, dim, bias=False)

        self.softmax = nn.Softmax()

    def forward(self, h_t, h, attn_history=None, mask=None, debug=False):
        """
        Args:
            h_t : [bs x dim]
            h   : [n x bs x dim]
            attn_history: None or [(t-1) x bs x n]
        Returns:
            C_t :  [bs x n]
            alpha: [bs x dim]
            attn_history: [t x bs x n]
        """
        bs, dim = h_t.size()
        n, _bs, _dim = h.size()
        assert (_bs, _dim) == (bs, dim)
        if attn_history is not None:
            _t, __bs, _n = attn_history.size()
            assert (__bs, _n) == (_bs, n)

        _h_t = self.linear(h_t).unsqueeze(1)
        _h = h.view(n, bs, dim)

        # e_t = [bs, 1, dim] bmm [bs, dim, n] = [bs, n] (after squeeze)
        E = _h_t.bmm(_h.transpose(0, 1).transpose(1, 2)).squeeze(1)

        next_attn_history = None
        alpha = None
        if self.temporal:
            if attn_history is None:
                next_attn_history = E.unsqueeze(0)
            else:
                # we manually computes the softmax because it must not include
                # E itself. We substract the maximum value in order to ensure
                # numerical stability
                next_attn_history = torch.cat([attn_history,
                                               E.unsqueeze(0)], 0)
                M = next_attn_history.max(0)[0]
                E = (E - M).exp() / (attn_history - M).exp().sum(0)
                S = E.sum(1)
                alpha = E / S.unsqueeze(1)

        if alpha is None:
            alpha = self.softmax(E)

        C_t = alpha.unsqueeze(1).bmm(_h.transpose(0, 1)).squeeze(1)

        if self.temporal:
            return C_t, alpha, next_attn_history
        return C_t, alpha


class PointerGenerator(CopyGenerator):
    def __init__(self, opt, tgt_vocab, embeddings):
        super(PointerGenerator, self).__init__(opt.rnn_size, tgt_vocab)
        self.input_size = opt.rnn_size * 3
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


class ReinforcedDecoder(_Module):
    def __init__(self, opt, embeddings, dec_attn=True,
                 exp_bias_reduction=0.25, bidirectional_encoder=False):
        """
        Implementation of a decoder following Paulus et al., (2017)
        By default, we refer to this paper when mentioning a section


        Args:
            opt:
            embeddings: target embeddings
            dec_attn: boolean, use decoder intra attention or not (sect. 2.2)
            exp_bias_reduction: float in [0, 1], exposure bias reduction by
                                feeding predicted token with a given
                                probability as mentionned in sect. 6.1
            bidirectional_encoder
        """
        super(ReinforcedDecoder, self).__init__(opt)
        self.embeddings = embeddings
        W_emb = embeddings.weight
        self.tgt_vocab_size, self.input_size = W_emb.size()
        self.dim = opt.rnn_size

        # TODO use parameter instead of hardcoding nlayer
        self.rnn = onmt.modules.StackedLSTM(1, self.input_size,
                                            self.dim, opt.dropout)

        self.enc_attn = IntraAttention(opt, self.dim, temporal=True)

        self.dec_attn = None
        if dec_attn:
            self.dec_attn = IntraAttention(opt, self.dim)

        self.pad_id = embeddings.word_padding_idx
        self.exp_bias_reduction = exp_bias_reduction

        # For compatibility reasons, TODO refactor
        self.hidden_size = self.dim
        self.decoder_type = "reinforced"
        self.bidirectional_encoder = bidirectional_encoder

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, src, context, enc_hidden):
        """
        Args:
            src: For compatibility reasons.......

        """
        if isinstance(enc_hidden, tuple):  # GRU
            return onmt.Models.RNNDecoderState(
                self.hidden_size,
                tuple([self._fix_enc_hidden(enc_hidden[i])
                       for i in range(len(enc_hidden))]))
        else:  # LSTM
            return onmt.Models.RNNDecoderState(
                self.hidden_size, self._fix_enc_hidden(enc_hidden))

    def forward(self, inputs, src, h_e, state, batch,
                loss_compute=None, tgt=None, generator=None,
                hd_history=None, attn_hist=None, ret_hists=False,
                sampling=False):
        """
        Args:
            inputs (LongTensor): [tgt_len x bs]
            src (LongTensor): [src_len x bs x 1]
            h_e (FloatTensor): [src_len x bs x dim]
            state: onmt.Models.DecoderState
            tgt (LongTensor): [tgt_len x bs]

        Returns:
            stats:
            state:
            scores:
            attns:
            hd_history: memory for decoder intra attention
            attn_hist: memory for temporal attention

        """
        dim = self.dim
        src_len, bs, _ = list(src.size())
        input_size, _bs = list(inputs.size())
        assert bs == _bs, "bs does not match %d, %d" % (bs, _bs)

        if self.training:
            assert tgt is not None
        if tgt is not None:
            assert loss_compute is not None
            if generator is not None:
                print("[WARNING] Parameter 'generator' should not "
                      + "be set at training time")
        else:
            assert generator is not None

        # src as [bs x src_len]
        src = src.transpose(0, 1).squeeze(2).contiguous()

        stats = onmt.Statistics()
        hidden = state.hidden
        loss = None
        scores, attns, dec_attns, outputs = [], [], [], []
        preds = []
        inputs_t = inputs[0, :]

        for t in range(input_size):
            # Embedding & intra-temporal attention on source
            emb_t = self.embeddings(inputs_t.view(1, -1, 1)).squeeze(0)

            hd_t, hidden = self.rnn(emb_t, hidden)

            c_e, alpha_e, attn_hist = self.enc_attn(hd_t,
                                                    h_e,
                                                    attn_history=attn_hist)

            # Intra-decoder Attention
            if self.dec_attn is None or hd_history is None:
                # no decoder intra attn at first step
                cd_t = self.mkvar(torch.zeros([bs, dim]))
                alpha_d = cd_t
                hd_history = hd_t.unsqueeze(0)
            else:
                cd_t, alpha_d = self.dec_attn(hd_t, hd_history)
                hd_history = torch.cat([hd_history, hd_t.unsqueeze(0)], dim=0)

            # Prediction - Computing Loss
            if tgt is not None:
                output = torch.cat([hd_t, c_e, cd_t], dim=1)
                if sampling:
                    prediction_type = "sample"
                    # TODO here 0 and 3 are hardcoded
                    continue_gen = (inputs_t.ne(3) * inputs_t.ne(0))
                    tgt_t = continue_gen.long()
                    align = torch.autograd.Variable(
                        torch.zeros([bs]).long().cuda())
                else:
                    tgt_t = tgt[t, :]
                    prediction_type = "greedy"
                    align = batch.alignment[t, :].contiguous()

                loss_t, pred_t, stats_t = loss_compute.compute_loss(
                    batch,
                    output,
                    tgt_t,
                    copy_attn=alpha_e,
                    align=align,
                    src=src,
                    prediction_type=prediction_type)
                outputs += [output]
                attns += [alpha_e]
                preds += [pred_t]

                stats.update(stats_t)
                loss = loss + loss_t if loss is not None else loss_t

            else:
                # In translation case we just want scores
                # prediction itself will be done with beam search
                output = torch.cat([hd_t, c_e, cd_t], dim=1)
                scores_t = generator(output, alpha_e, batch.src_map)
                scores += [scores_t]
                attns += [alpha_e]
                dec_attns += [alpha_d]

            if sampling:
                # the sampling mode correspond to generating y^s_t as
                # described in sect. 3.2
                inputs_t = preds[-1]

            elif t < input_size - 1:
                if self.training:
                    # Exposure bias reduction by feeding predicted token
                    # with a 0.25 probability as mentionned in sect. 6.1

                    _pred_t = preds[-1].clone()
                    _pred_t = loss_compute.remove_oov(_pred_t)
                    exposure_mask = self.mkvar(
                        torch.rand([bs]).lt(self.exp_bias_reduction).long())
                    inputs_t = exposure_mask * _pred_t.long()
                    inputs_t += (1 - exposure_mask.float()).long() \
                        * inputs[t+1, :]

                else:
                    inputs_t = inputs[t+1, :]

        state.update_state(hidden, None, None)
        if not ret_hists:
            return loss, stats, state, scores, attns, preds
        return stats, state, scores, attns, hd_history, attn_hist


class ReinforcedModel(onmt.Models.NMTModel):
    def __init__(self, encoder, decoder, multigpu=False, gamma=0.9984):
        """
        Args:
            encoder:
            decoder:
            multigpu: not sure why its here
            gamma: in [0;1] weight between ML and RL
                   loss = gamma * loss_rl + (1 - gamma) * loss_ml
                   (see Paulus et al 2017, sect. 3.3)
        """
        super(ReinforcedModel, self).__init__(encoder, decoder)
        self.rouge = RougeScorer()
        self.gamma = gamma
        self.model_type = "Reinforced"

    def forward(self, src, tgt, src_lengths, batch, loss_compute,
                dec_state=None):
        """
        Args:
            src:
            tgt:
            dec_state: A decoder state object
        """
        n_feats = tgt.size(2)
        assert n_feats == 1, "Reinforced model does not handle features"
        tgt = tgt.squeeze(2)
        enc_hidden, enc_out = self.encoder(src, src_lengths)

        enc_state = self.decoder.init_decoder_state(src=None,
                                                    enc_hidden=enc_hidden,
                                                    context=enc_out)
        state = enc_state if dec_state is None else dec_state

        loss, stats, hidden, _, _, preds = self.decoder(tgt[:-1],
                                                        src,
                                                        enc_out,
                                                        state, batch,
                                                        loss_compute,
                                                        tgt=tgt[1:])

        if self.gamma > 0:
            loss2, stats2, hidden2, _, _, preds2 = self.decoder(tgt[:-1],
                                                                src,
                                                                enc_out,
                                                                state,
                                                                batch,
                                                                loss_compute,
                                                                tgt=tgt[1:],
                                                                sampling=True)

            sample_preds = torch.stack(preds2, 1)
            greedy_preds = torch.stack(preds, 1)
            metric = self.rouge.score(sample_preds, greedy_preds, tgt[1:].t())
            metric = torch.autograd.Variable(metric).cuda()
            rl_loss = (loss2 * metric).sum()

            loss = (self.gamma * rl_loss) - ((1 - self.gamma * loss))
        return loss, stats, state


class DummyGenerator:
    """Hacky way to ensure compatibility
    """

    def dummy_pass(self, *args, **kwargs):
        pass

    def __init__(self, *args, **kwargs):
        self.state_dict = self.dummy_pass
        self.cpu = self.dummy_pass
        self.cuda = self.dummy_pass
        self.__call__ = self.dummy_pass
        self.load_state_dict = self.dummy_pass

    def __getattr__(self, attr):
        class DummyCallableObject:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                pass

        return DummyCallableObject()
