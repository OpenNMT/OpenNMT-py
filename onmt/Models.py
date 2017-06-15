import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


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
        self.layers = opt.layers

        # Use a bidirectional model.
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0

        # Size of the encoder RNN.
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()

        # Word embeddings.
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)

        # Feature embeddings.
        if feature_dicts:
            self.feature_luts = nn.ModuleList([
                nn.Embedding(feature_dict.size(),
                             opt.feature_vec_size,
                             padding_idx=onmt.Constants.PAD)
                for feature_dict in feature_dicts])

            # MLP on features and words.
            self.activation = nn.ReLU()
            self.linear = nn.Linear(input_size +
                                    len(feature_dicts) * opt.feature_vec_size,
                                    self.hidden_size)
            input_size = self.hidden_size

        else:
            self.feature_luts = nn.ModuleList([])

        # The Encoder RNN.
        self.encoder_layer = opt.encoder_layer if "encoder_layer" in opt else ""
        self.rnn = nn.LSTM(input_size, self.hidden_size,
                           num_layers=opt.layers,
                           dropout=opt.dropout,
                           bidirectional=opt.brnn)

        
    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def _embed(self, src_input):
        """
        Embed the words or utilize features and MLP.
        
        Args:
            src_input (LongTensor): len x batch x nfeat

        Return:
            emb (FloatTensor): len x batch x input_size
        """
        if self.feature_luts:
            word = self.word_lut(src_input[:, :, 0])
            features = [feature_lut(src_input[:, :, j+1])
                        for j, feature_lut in enumerate(self.feature_luts)]
            # Concat feature and word embeddings.
            emb = torch.cat([word] + features, -1)

            # Apply one MLP layer.
            emb2 = self.activation(self.linear(emb.view(-1, emb.size(-1))))
            emb = emb2.view(emb.size(0), emb.size(1), -1)
        else:
            emb = self.word_lut(src_input[:, :, 0])
        return emb

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat
            lengths (LongTensor): batch
            hidden: Initial hidden state.

        Returns:
            hidden_t (FloatTensor): Pair of layers x batch x rnn_size - final Encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        if lengths:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.data.view(-1).tolist()
            pre_emb = self._embed(input)
            emb = pack(pre_emb, lengths)
        else:
            emb = self._embed(input)
            pre_emb = emb
            
        if self.encoder_layer == "mean":
            mean = pre_emb.mean(0).view(1, pre_emb.size(1), pre_emb.size(2)) \
                   .expand(self.layers, pre_emb.size(1), pre_emb.size(2))

            return (mean, mean), pre_emb
        
        outputs, hidden_t = self.rnn(emb, hidden)
        if lengths:
            outputs = unpack(outputs)[0]
        return hidden_t, outputs


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM. 
    Needed for the decoder, because we do input feeding.
    """
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
    """
    Decoder + Attention recurrent neural network.
    """
    
    def __init__(self, opt, dicts):
        """
        Args:
            opt: model options
            dicts: Target `Dict` object
        """
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
        self.dropout = nn.Dropout(opt.dropout)
        self.hidden_size = opt.rnn_size

        # Std attention layer.
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        
        # Separate Copy Attention.
        self._copy = False
        if opt.copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(opt.rnn_size)
            self._copy = True
            
    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context, init_feed):
        """
        Forward through the decoder.

        Args:
            input (LongTensor):  (len x batch) -- Input tokens
            hidden (FloatTensor): tuple (1 x batch x rnn_size) -- Init hidden state
            context (FloatTensor):  (src_len x batch x rnn_size)  -- Memory bank
            init_feed (FloatTensor): tuple(batch_size) (rnn_size) -- Init input feed

        Returns:
            outputs (FloatTensor): (len x batch x rnn_size)
            hidden_t (FloatTensor): Last hidden state tuple (1 x batch x rnn_size)
            attns (FloatTensor): Dictionary of (src_len x batch)
        """
        emb = self.word_lut(input)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
            
        output = init_feed
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.t())

            output = self.dropout(output)
            outputs += [output]
            attns["std"] += [attn]

            # COPY
            if self._copy:
                _, copy_attn = self.copy_attn(output, context.t())
                attns["copy"] += [copy_attn]
        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])
        return outputs, hidden, attns


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def make_init_decoder_output(self, context):
        """
        Constructs a vector that can be used to initialize the
        decoder context.

        Args:
            context: FloatTensor (batch x src_len x renn_size)

        Returns:
            decoder_output: FloatTensor variable (batch x hidden)
        """
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim
        We need to convert it to layers x batch x (directions*dim)
        """
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, input, dec_hidden=None):
        """
        Args:
            input: A `Batch` object.
            dec_hidden (FloatTensor): tuple (1 x batch x rnn_size) -- Init hidden state

        Returns:
            outputs (FloatTensor): (len x batch x rnn_size) -- Decoder outputs.
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hidden (FloatTensor): tuple (1 x batch x rnn_size) -- Init hidden state
        """
        src = input.src
        tgt = input.tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, input.lengths)
        init_output = self.make_init_decoder_output(context)

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
        out, dec_hidden, attns = self.decoder(tgt,
                                              enc_hidden if dec_hidden is None
                                              else dec_hidden,
                                              context,
                                              init_output)

        return out, attns, dec_hidden
