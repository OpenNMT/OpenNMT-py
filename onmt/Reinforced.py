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
from onmt.modules import IntraAttention


class RougeScorer:
    def __init__(self):
        import rouge as R
        self.rouge = R.Rouge(stats=["f"], metrics=[
                             "rouge-1", "rouge-2", "rouge-l"])

    def _score(self, hyps, refs):
        scores = self.rouge.get_scores(hyps, refs)
        # NOTE: here we use score = r1 * r2 * rl
        #       I'm not sure how relevant it is
        metric_weight = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 1}

        scores = [sum([seq[metric]['f'] * metric_weight[metric]
                       for metric in seq.keys()])
                  for seq in scores]
        return scores

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


class ReinforcedDecoder(nn.Module):
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

        self.enc_attn = IntraAttention(self.dim, temporal=True)

        self.dec_attn = None
        if dec_attn:
            self.dec_attn = IntraAttention(self.dim)

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
    def __init__(self, encoder, decoder, gamma=0.9984):
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

        ml_loss, stats, hidden, _, _, ml_preds = self.decoder(tgt[:-1],
                                                              src,
                                                              enc_out,
                                                              state,
                                                              batch,
                                                              loss_compute,
                                                              tgt=tgt[1:])

        if self.gamma > 0:
            rl_loss, stats2, hidden2, _, _, rl_preds = \
                self.decoder(tgt[:-1],
                             src,
                             enc_out,
                             state,
                             batch,
                             loss_compute,
                             tgt=tgt[1:],
                             sampling=True)

            sample_preds = torch.stack(rl_preds, 1)
            greedy_preds = torch.stack(ml_preds, 1)
            metric = self.rouge.score(sample_preds, greedy_preds, tgt[1:].t())
            metric = torch.autograd.Variable(metric).cuda()

            rl_loss = (rl_loss * metric).sum()
            loss = (self.gamma * rl_loss) - ((1 - self.gamma * ml_loss))
        return loss, stats, state
