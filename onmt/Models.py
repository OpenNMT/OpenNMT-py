from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt
import onmt.modules
from onmt.modules import aeq
from onmt.modules.Gate import ContextGateFactory
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class Embeddings(nn.Module):

    def __init__(
            self, dicts, feature_dicts,
            word_vec_size, pre_word_vecs, fix_word_vecs,
            feat_merge, feat_vec_size, feat_vec_exponent,
            position_encoding, gpus, dropout):

        super(Embeddings, self).__init__()

        self.positional_encoding = position_encoding
        if self.positional_encoding:
            self.pe = self.make_positional_encodings(word_vec_size, 5000)
            if len(gpus) > 0:
                self.pe.cuda()
            self.dropout = nn.Dropout(p=dropout)

        self.feat_merge = feat_merge

        vocab_sizes = [dicts.size()]
        emb_sizes = [word_vec_size]
        if feature_dicts:
            vocab_sizes.extend(feat_dict.size() for feat_dict in feature_dicts)
            if feat_merge == 'concat':
                # Derive embedding sizes from each feature's vocab size
                emb_sizes.extend([int(feat_dict.size() ** feat_vec_exponent)

                                  for feat_dict in feature_dicts])
            elif feat_merge == 'sum':
                # All embeddings to be summed must be the same size
                emb_sizes.extend([word_vec_size] * len(feature_dicts))

            else:
                # mlp feature merge
                emb_sizes.extend([feat_vec_size] * len(feature_dicts))
                # apply a layer of mlp to get it down to the correct dim
                self.mlp = nn.Sequential(onmt.modules.BottleLinear(
                                        sum(emb_sizes),
                                        word_vec_size),
                                        nn.ReLU())

        self.emb_luts = nn.ModuleList([
                            nn.Embedding(vocab, dim,
                                         padding_idx=onmt.Constants.PAD)
                            for vocab, dim in zip(vocab_sizes, emb_sizes)])
        if pre_word_vecs:
            self._load_pretrained_vectors(pre_word_vecs)
        if fix_word_vecs:
            self.word_lut.weight.requires_grad = False

    @property
    def word_lut(self):
        return self.emb_luts[0]

    @property
    def embedding_size(self):
        """
        Returns sum of all feature dimensions if the merge action is concat.
        Otherwise, returns word vector size.
        """
        if self.feat_merge == 'concat':
            return sum(emb_lut.embedding_dim
                       for emb_lut in self.emb_luts.children())
        else:
            return self.word_lut.embedding_dim

    def make_positional_encodings(self, dim, max_len):
        pe = torch.arange(0, max_len).unsqueeze(1).expand(max_len, dim)
        div_term = 1 / torch.pow(10000, torch.arange(0, dim * 2, 2) / dim)
        pe = pe * div_term.expand_as(pe)
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(1)

    def _load_pretrained_vectors(self, emb_file):
        pretrained = torch.load(emb_file)
        self.word_lut.weight.data.copy_(pretrained)

    def merge(self, features):
        if self.feat_merge == 'concat':
            return torch.cat(features, 2)
        elif self.feat_merge == 'sum':
            return sum(features)
        else:
            return self.mlp(torch.cat(features, 2))

    def forward(self, src_input):
        """
        Return the embeddings for words, and features if there are any.
        Args:
            src_input (LongTensor): len x batch x nfeat
        Return:
            emb (FloatTensor): len x batch x self.embedding_size
        """
        in_length, in_batch, nfeat = src_input.size()
        aeq(nfeat, len(self.emb_luts))

        if len(self.emb_luts) == 1:
            emb = self.word_lut(src_input.squeeze(2))
        else:
            feat_inputs = (feat.squeeze(2)
                           for feat in src_input.split(1, dim=2))
            features = [lut(feat)
                        for lut, feat in zip(self.emb_luts, feat_inputs)]
            emb = self.merge(features)

        if self.positional_encoding:
            emb = emb + Variable(self.pe[:emb.size(0), :1, :emb.size(2)]
                                 .expand_as(emb))
            emb = self.dropout(emb)

        out_length, out_batch, emb_size = emb.size()
        aeq(in_length, out_length)
        aeq(in_length, out_length)
        aeq(emb_size, self.embedding_size)

        return emb


class Encoder(nn.Module):
    """
    Encoder recurrent neural network.
    """
    def __init__(self, opt, dicts, feature_dicts=None):
        """
        Args:
            opt: Model options.
            dicts (`Dict`): The src dictionary
            features_dicts (`[Dict]`): List of src feature dictionaries.
        """
        # Number of rnn layers.
        self.layers = opt.enc_layers

        # Use a bidirectional model.
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0

        # Size of the encoder RNN.
        self.hidden_size = opt.rnn_size // self.num_directions

        super(Encoder, self).__init__()
        self.embeddings = Embeddings(
            dicts, feature_dicts,
            opt.word_vec_size, opt.pre_word_vecs_enc, opt.fix_word_vecs_enc,
            opt.feat_merge, opt.feat_vec_size, opt.feat_vec_exponent,
            opt.position_encoding, opt.gpus, opt.dropout)

        input_size = self.embeddings.embedding_size

        # The Encoder RNN.
        self.encoder_type = opt.encoder_type
        pad_id = dicts.stoi[onmt.IO.PAD_WORD]

        if self.encoder_type == "transformer":
            self.transformer = nn.ModuleList(
                [onmt.modules.TransformerEncoder(self.hidden_size, opt, pad_id)
                 for i in range(opt.enc_layers)])
        else:
            self.rnn = getattr(nn, opt.rnn_type)(
                 input_size, self.hidden_size,
                 num_layers=opt.enc_layers,
                 dropout=opt.dropout,
                 bidirectional=opt.brnn)

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat
            lengths (LongTensor): batch
            hidden: Initial hidden state.

        Returns:
            hidden_t (FloatTensor): Pair of layers x batch x rnn_size - final
                                    Encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        # CHECKS
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)
        # END CHECKS

        emb = self.embeddings(input)
        s_len, n_batch, vec_size = emb.size()

        if self.encoder_type == "mean":
            # No RNN, just take mean as final state.
            mean = emb.mean(0) \
                   .expand(self.layers, n_batch, vec_size)
            return (mean, mean), emb

        elif self.encoder_type == "transformer":
            # Self-attention tranformer.
            out = emb.transpose(0, 1).contiguous()
            for i in range(self.layers):
                out = self.transformer[i](out, input[:, :, 0].transpose(0, 1))
            return Variable(emb.data), out.transpose(0, 1).contiguous()
        else:
            # Standard RNN encoder.
            packed_emb = emb
            if lengths is not None:
                # Lengths data is wrapped inside a Variable.
                lengths = lengths.view(-1).tolist()
                packed_emb = pack(emb, lengths)
            outputs, hidden_t = self.rnn(packed_emb, hidden)
            if lengths:
                outputs = unpack(outputs)[0]
            return hidden_t, outputs


class Decoder(nn.Module):
    """
    Decoder + Attention recurrent neural network.
    """

    def __init__(self, opt, dicts):
        """
        Args:
            opt: model options
            dicts: Target `Dict` object
        """
        self.layers = opt.dec_layers
        self.decoder_type = opt.decoder_type
        self._coverage = opt.coverage_attn
        self.hidden_size = opt.rnn_size
        self.input_feed = opt.input_feed
        input_size = opt.tgt_word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.embeddings = Embeddings(
            dicts, None,
            opt.word_vec_size, opt.pre_word_vecs_dec, opt.fix_word_vecs_dec,
            opt.feat_merge, opt.feat_vec_size, opt.feat_vec_exponent,
            opt.position_encoding, opt.gpus, opt.dropout)

        pad_id = dicts.stoi[onmt.IO.PAD_WORD]
        if self.decoder_type == "transformer":
            self.transformer = nn.ModuleList(
                [onmt.modules.TransformerDecoder(self.hidden_size, opt, pad_id)
                 for _ in range(opt.dec_layers)])
        else:
            if self.input_feed:
                if opt.rnn_type == "LSTM":
                    stackedCell = onmt.modules.StackedLSTM
                else:
                    stackedCell = onmt.modules.StackedGRU
                self.rnn = stackedCell(opt.dec_layers, input_size,
                                       opt.rnn_size, opt.dropout)
            else:
                self.rnn = getattr(nn, opt.rnn_type)(
                     input_size, opt.rnn_size,
                     num_layers=opt.dec_layers,
                     dropout=opt.dropout
                )
            self.context_gate = None
            if opt.context_gate is not None:
                self.context_gate = ContextGateFactory(
                    opt.context_gate, input_size,
                    opt.rnn_size, opt.rnn_size,
                    opt.rnn_size
                )

        self.dropout = nn.Dropout(opt.dropout)

        # Std attention layer.
        self.attn = onmt.modules.GlobalAttention(
            opt.rnn_size,
            coverage=self._coverage,
            attn_type=opt.global_attention)

        # Separate Copy Attention.
        self._copy = False
        if opt.copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                opt.rnn_size, attn_type=opt.global_attention)
            self._copy = True

    def forward(self, input, src, context, state):
        """
        Forward through the decoder.

        Args:
            input (LongTensor):  (len x batch) -- Input tokens
            src (LongTensor)
            context:  (src_len x batch x rnn_size)  -- Memory bank
            state: an object initializing the decoder.

        Returns:
            outputs: (len x batch x rnn_size)
            final_states: an object of the same form as above
            attns: Dictionary of (src_len x batch)
        """
        # CHECKS
        t_len, n_batch = input.size()
        s_len, n_batch_, _ = src.size()
        s_len_, n_batch__, _ = context.size()
        aeq(n_batch, n_batch_, n_batch__)
        # aeq(s_len, s_len_)
        # END CHECKS
        if self.decoder_type == "transformer":
            if state.previous_input:
                input = torch.cat([state.previous_input.squeeze(2), input], 0)

        emb = self.embeddings(input.unsqueeze(2))

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []

        # Setup the different types of attention.
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        if self.decoder_type == "transformer":
            # Tranformer Decoder.
            assert isinstance(state, TransformerDecoderState)
            output = emb.transpose(0, 1).contiguous()
            src_context = context.transpose(0, 1).contiguous()
            for i in range(self.layers):
                output, attn \
                    = self.transformer[i](output, src_context,
                                          src[:, :, 0].transpose(0, 1),
                                          input.transpose(0, 1))
            outputs = output.transpose(0, 1).contiguous()
            if state.previous_input:
                outputs = outputs[state.previous_input.size(0):]
                attn = attn[:, state.previous_input.size(0):].squeeze()
                attn = torch.stack([attn])
            attns["std"] = attn
            if self._copy:
                attns["copy"] = attn
            state = TransformerDecoderState(input.unsqueeze(2))
        elif self.input_feed:
            assert isinstance(state, RNNDecoderState)
            output = state.input_feed.squeeze(0)
            hidden = state.hidden
            # CHECKS
            n_batch_, _ = output.size()
            aeq(n_batch, n_batch_)
            # END CHECKS

            coverage = state.coverage.squeeze(0) \
                if state.coverage is not None else None

            # Standard RNN decoder.
            for i, emb_t in enumerate(emb.split(1)):
                emb_t = emb_t.squeeze(0)
                if self.input_feed:
                    emb_t = torch.cat([emb_t, output], 1)

                rnn_output, hidden = self.rnn(emb_t, hidden)
                attn_output, attn = self.attn(rnn_output,
                                              context.transpose(0, 1))
                if self.context_gate is not None:
                    output = self.context_gate(
                        emb_t, rnn_output, attn_output
                    )
                    output = self.dropout(output)
                else:
                    output = self.dropout(attn_output)
                outputs += [output]
                attns["std"] += [attn]

                # COVERAGE
                if self._coverage:
                    coverage = coverage + attn \
                               if coverage is not None else attn
                    attns["coverage"] += [coverage]

                # COPY
                if self._copy:
                    _, copy_attn = self.copy_attn(output,
                                                  context.transpose(0, 1))
                    attns["copy"] += [copy_attn]
            state = RNNDecoderState(hidden, output.unsqueeze(0),
                                    coverage.unsqueeze(0)
                                    if coverage is not None else None)
            outputs = torch.stack(outputs)
            for k in attns:
                attns[k] = torch.stack(attns[k])
        else:
            assert isinstance(state, RNNDecoderState)
            assert emb.dim() == 3

            assert not self._coverage
            assert state.coverage is None

            # TODO: copy
            assert not self._copy

            hidden = state.hidden
            rnn_output, hidden = self.rnn(emb, hidden)

            # CHECKS
            t_len_, n_batch_, _ = rnn_output.size()
            aeq(n_batch, n_batch_)
            aeq(t_len, t_len_)
            # END CHECKS

            attn_outputs, attn_scores = self.attn(
                rnn_output.transpose(0, 1).contiguous(),    # (batch, t_len, d)
                context.transpose(0, 1)                     # (batch, s_len, d)
            )

            if self.context_gate is not None:
                outputs = self.context_gate(
                    emb.view(-1, emb.size(2)),
                    rnn_output.view(-1, rnn_output.size(2)),
                    attn_outputs.view(-1, attn_outputs.size(2))
                )
                outputs = outputs.view(t_len, n_batch, self.hidden_size)
                outputs = self.dropout(outputs)
            else:
                outputs = self.dropout(attn_outputs)        # (t_len, batch, d)
            state = RNNDecoderState(hidden, outputs[-1].unsqueeze(0), None)
            attns["std"] = attn_scores

        return outputs, state, attns


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim
        We need to convert it to layers x batch x (directions*dim)
        """
        if self.encoder.num_directions == 2:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, context, enc_hidden):
        if self.decoder.decoder_type == "transformer":
            return TransformerDecoderState()
        elif isinstance(enc_hidden, tuple):
            dec = RNNDecoderState(tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:
            dec = RNNDecoderState(self._fix_enc_hidden(enc_hidden))
        dec.init_input_feed(context, self.decoder.hidden_size)
        return dec

    def forward(self, src, tgt, lengths, dec_state=None):
        """
        Args:
            src, tgt, lengths
            dec_state: A decoder state object

        Returns:
            outputs (FloatTensor): (len x batch x rnn_size) -- Decoder outputs.
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hidden (FloatTensor): tuple (1 x batch x rnn_size)
                                      Init hidden state
        """
        src = src
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, lengths)
        enc_state = self.init_decoder_state(context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, src, context,
                                             enc_state if dec_state is None
                                             else dec_state)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state


class DecoderState(object):
    def detach(self):
        for h in self.all:
            if h is not None:
                h.detach_()

    def repeatBeam_(self, beamSize):
        self._resetAll([Variable(e.data.repeat(1, beamSize, 1))
                        for e in self.all])

    def beamUpdate_(self, idx, positions, beamSize):
        for e in self.all:
            a, br, d = e.size()
            sentStates = e.view(a, beamSize, br // beamSize, d)[:, :, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, rnnstate, input_feed=None, coverage=None):
        # all objects are X x batch x dim
        # or X x (beam * sent) for beam search
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage
        self.all = self.hidden + (self.input_feed,)

    def init_input_feed(self, context, rnn_size):
        batch_size = context.size(1)
        h_size = (batch_size, rnn_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)
        self.all = self.hidden + (self.input_feed,)

    def _resetAll(self, all):
        vars = [Variable(a.data if isinstance(a, Variable) else a,
                         volatile=True) for a in all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]
        self.all = self.hidden + (self.input_feed,)


class TransformerDecoderState(DecoderState):
    def __init__(self, input=None):
        # all objects are X x batch x dim
        # or X x (beam * sent) for beam search
        self.previous_input = input
        self.all = (self.previous_input,)

    def _resetAll(self, all):
        vars = [(Variable(a.data if isinstance(a, Variable) else a,
                          volatile=True))
                for a in all]
        self.previous_input = vars[0]
        self.all = (self.previous_input,)

    def repeatBeam_(self, beamSize):
        pass


def make_base_model(opt, model_opt, fields, cuda, checkpoint=None):
    # HACK: collect source feature vocabs.
    feature_vocabs = []
    for j in range(100):
        key = "src_feat_" + str(j)
        if key not in fields:
            break
        feature_vocabs.append(fields[key].vocab)

    if model_opt.model_type == "text":
        encoder = Encoder(model_opt, fields["src"].vocab,
                          feature_vocabs)
    elif model_opt.model_type == "img":
        encoder = onmt.modules.ImageEncoder(model_opt)
    else:
        assert False, ("Unsupported model type %s"
                       % (model_opt.model_type))

    decoder = onmt.Models.Decoder(
        model_opt, fields["tgt"].vocab)
    model = onmt.Models.NMTModel(encoder, decoder)

    if not model_opt.copy_attn:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields["tgt"].vocab)),
            nn.LogSoftmax())
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = onmt.modules.CopyGenerator(model_opt, fields["src"].vocab,
                                               fields["tgt"].vocab)

    if checkpoint is not None:
        print('Loading model')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])

    if cuda:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()
    model.generator = generator
    return model
