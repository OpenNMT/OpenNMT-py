import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class Encoder(nn.Module):
    def __init__(self, opt, dicts, feature_dicts=None):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        if feature_dicts:
            self.feature_luts = nn.ModuleList([
                nn.Embedding(feature_dict.size(),
                             opt.feature_vec_size,
                             padding_idx=onmt.Constants.PAD)
                for feature_dict in feature_dicts])

            self.linear = nn.Linear(input_size + len(feature_dicts) * opt.feature_vec_size,
                                    self.hidden_size)
            input_size = self.hidden_size
            # 
        else:
            self.feature_luts = nn.ModuleList([])
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size,
                           num_layers=opt.layers,
                           dropout=opt.dropout,
                           bidirectional=opt.brnn)
        self.activation = nn.ReLU()

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
        def embed(input):
            if self.feature_luts:
                word = self.word_lut(input[:, :, 0])
                features = [feature_lut(input[:, :, j+1])
                            for j, feature_lut in enumerate(self.feature_luts)]
                emb = torch.cat([word] +  features, -1)
                emb2 = self.activation(self.linear(emb.view(-1, emb.size(-1))))
                emb = emb2.view(emb.size(0), emb.size(1), -1)                
            else:
                emb = self.word_lut(input)
            return emb
            
        if isinstance(input, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = input[1].data.view(-1).tolist()
            pre_emb = embed(input[0])
            emb = pack(pre_emb, lengths)
        else:
            emb = embed(input)

        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
            
        # return hidden_t, outputs
        return hidden_t, pre_emb #tuple([pre_emb[i] for i in range(pre_emb.size(0))])


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        self.rnn = StackedLSTM(opt.layers, input_size,
                               opt.rnn_size, opt.dropout)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.copy_attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size
        self.trunc_bptt_cutoff = opt.trunc_bptt_cutoff

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context, init_output):
        """
        context : batch x src_size x hidden
        """
        emb = self.word_lut(input)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        attns = []
        output = init_output
        copy_output = init_output
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            if self.trunc_bptt_cutoff \
               and i != 0 and i % self.trunc_bptt_cutoff == 0:
                hidden = [h.detach() for h in hidden]
            output, attn = self.attn(output, context.t())
            copy_output, copy_attn = self.copy_attn(output, context.t())
            output = self.dropout(output)
            outputs += [output]
            attns += [copy_attn]
        outputs = torch.stack(outputs)
        attns = torch.stack(attns)
        return outputs, hidden, attns


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, input, dec_hidden=None):
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src)
        init_output = self.make_init_decoder_output(context)

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
        out, dec_hidden, _attn = self.decoder(tgt, 
                                              enc_hidden if dec_hidden is None else dec_hidden,
                                              context,
                                              init_output)

        return out, _attn, dec_hidden
