import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt
import onmt.modules
from onmt.modules.Gate import ContextGateFactory
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import math


class Embeddings(nn.Module):
    def __init__(self, opt, dicts, feature_dicts=None):
        self.positional_encoding = opt.__dict__.get("position_encoding", "")
        if self.positional_encoding:
            self.pe = self.make_positional_encodings(opt.word_vec_size, 5000).cuda()

        self.word_vec_size = opt.word_vec_size
        
        super(Embeddings, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        # Word embeddings.
        self.dropout = nn.Dropout(p=opt.dropout)

        # Feature embeddings.
        if feature_dicts:
            self.feature_luts = nn.ModuleList([
                nn.Embedding(feature_dict.size(),
                             opt.feature_vec_size,
                             padding_idx=onmt.Constants.PAD)
                for feature_dict in feature_dicts])

            # MLP on features and words.
            self.activation = nn.ReLU()
            self.linear = nn.Linear(opt.word_vec_size +
                                    len(feature_dicts) * opt.feature_vec_size,
                                    opt.word_vec_size)
        else:
            self.feature_luts = nn.ModuleList([])

    def make_positional_encodings(self, dim, max_len):
        pe = torch.FloatTensor(max_len, 1, dim).fill_(0)
        for i in range(dim):
            for j in range(max_len):
                k = float(j) / (10000.0 ** (2.0*i / float(dim)))
                pe[j, 0, i] = math.cos(k) if i % 2 == 1 else math.sin(k)
        return pe

    def load_pretrained_vectors(self, emb_file):
        if emb_file is not None:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, src_input):
        """
        Embed the words or utilize features and MLP.

        Args:
            src_input (LongTensor): len x batch x nfeat

        Return:
            emb (FloatTensor): len x batch x input_size
        """
        word = self.word_lut(src_input[:, :, 0])
        if self.feature_luts:
            features = [feature_lut(src_input[:, :, j+1])
                        for j, feature_lut in enumerate(self.feature_luts)]
            # Concat feature and word embeddings.
            emb = torch.cat([word] + features, -1)

            # Apply one MLP layer.
            emb2 = self.activation(self.linear(emb.view(-1, emb.size(-1))))
            emb = emb2.view(emb.size(0), emb.size(1), -1)
        else:
            emb = word

        if self.positional_encoding:
            emb = emb + Variable(self.pe[:emb.size(0), :1, :emb.size(2)]
                                 .expand_as(emb))
            emb = self.dropout(emb)
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
        self.layers = opt.layers

        # Use a bidirectional model.
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0

        # Size of the encoder RNN.
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.embeddings = Embeddings(opt, dicts, feature_dicts)
        
        # The Encoder RNN.
        self.encoder_layer = opt.__dict__.get("encoder_layer", "")

        if self.encoder_layer == "transformer":
            self.transformer = nn.ModuleList(
                [onmt.modules.TransformerEncoder(self.hidden_size, opt)
                 for i in range(opt.layers)])
        else:
            self.rnn = getattr(nn, opt.rnn_type)(
                 input_size, self.hidden_size,
                 num_layers=opt.layers,
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
        if lengths is not None:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.data.view(-1).tolist()
            pre_emb = self.embeddings(input)
            emb = pack(pre_emb, lengths)
        else:
            emb = self.embeddings(input)
            pre_emb = emb

        if self.encoder_layer == "mean":
            # Take the mean of vectors.
            mean = pre_emb.mean(0).view(1, pre_emb.size(1), pre_emb.size(2)) \
                   .expand(self.layers, pre_emb.size(1), pre_emb.size(2))
            return (mean, mean), pre_emb

        elif self.encoder_layer == "transformer":
            # Self-attention tranformer.
            mean = pre_emb.mean(0).view(1, pre_emb.size(1), pre_emb.size(2)) \
                   .expand(self.layers, pre_emb.size(1), pre_emb.size(2))
            out = pre_emb.transpose(0, 1).contiguous()
            for i in range(self.layers):
                out = self.transformer[i](out, input[:, :, 0])
            return (mean, mean), out.transpose(0, 1).contiguous()

        else:
            # Standard RNN encoder.
            outputs, hidden_t = self.rnn(emb, hidden)
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
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        self.decoder_layer = opt.__dict__.get("decoder_layer", "")
        self._coverage = opt.__dict__.get("coverage_attn", False)
        self.hidden_size = opt.rnn_size
        
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.embeddings = Embeddings(opt, dicts, None)

        if self.decoder_layer == "transformer":
            self.transformer = nn.ModuleList(
                [onmt.modules.TransformerDecoder(self.hidden_size, opt)
                 for _ in range(opt.layers)])
        else:
            if opt.rnn_type == "LSTM":
                stackedCell = onmt.modules.StackedLSTM
            else:
                stackedCell = onmt.modules.StackedGRU
            self.rnn = stackedCell(opt.layers, input_size,
                                   opt.rnn_size, opt.dropout)
            self.context_gate = None
            if opt.context_gate is not None:
                self.context_gate = ContextGateFactory(
                    opt.context_gate, opt.word_vec_size,
                    opt.rnn_size, opt.rnn_size, opt.rnn_size
                )

        self.dropout = nn.Dropout(opt.dropout)

        # Std attention layer.
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size, self._coverage)

        # Separate Copy Attention.
        self._copy = False
        if opt.__dict__.get("copy_attn", False):
            self.copy_attn = onmt.modules.GlobalAttention(opt.rnn_size)
            self._copy = True
            
    def forward(self, input, src, hidden, context, init_feed):
        """
        Forward through the decoder.

        Args:
            input (LongTensor):  (len x batch) -- Input tokens
            src (LongTensor)
            hidden: tuple (1 x batch x rnn_size) -- Init hidden state
            context:  (src_len x batch x rnn_size)  -- Memory bank
            init_feed: tuple(batch_size) (rnn_size) -- Init input feed

        Returns:
            outputs: (len x batch x rnn_size)
            hidden_t: Last hidden state tuple (1 x batch x rnn_size)
            attns: Dictionary of (src_len x batch)
        """

        # For transformer layers, hidden is the actual words
        has_transformer_hidden = (self.decoder_layer == "transformer"
                                  and hidden is not None)

        if has_transformer_hidden:
            input = torch.cat([hidden[0].squeeze(2), input], 0)

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

        output = init_feed
        coverage = None

        if self.decoder_layer == "transformer":
            # Tranformer Decoder. 
            src_context = context.transpose(0, 1).contiguous()
            for i in range(self.layers):
                output, attn = self.transformer[i](output, src_context,
                                                   src[:, :, 0], input)
            outputs = output.transpose(0, 1).contiguous()
            if has_transformer_hidden:
                outputs = outputs[hidden[0].size(0):]
                attn = attn[:, hidden[0].size(0):].squeeze()
                attn = torch.stack([attn])
            attns["std"] = attn
            if self._copy:
                attns["copy"] = attn
            hidden = (input.unsqueeze(2),)
        else:
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
                        emb_t.squeeze(0), rnn_output, attn_output
                    )
                    output = self.dropout(output)
                else:
                    output = self.dropout(attn_output)
                outputs += [output]

                if self._coverage:
                    if coverage:
                        coverage = coverage + attn
                    else:
                        coverage = attn
                    attns["coverage"] += [coverage]
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

    def setup_decoder(self, enc_hidden):
        if self.decoder.decoder_layer == "transformer":
            return None
        elif isinstance(enc_hidden, tuple):
            return tuple([self._fix_enc_hidden(enc_hidden[i])
                          for i in range(len(enc_hidden))])
        else:
            return self._fix_enc_hidden(enc_hidden)

    def forward(self, input, dec_hidden=None):
        """
        Args:
            input: A `Batch` object.
            dec_hidden (FloatTensor): tuple (1 x batch x rnn_size)
                                      Init hidden state

        Returns:
            outputs (FloatTensor): (len x batch x rnn_size) -- Decoder outputs.
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hidden (FloatTensor): tuple (1 x batch x rnn_size)
                                      Init hidden state
        """
        src = input.src
        tgt = input.tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, input.lengths)
        init_output = self.make_init_decoder_output(context)

        enc_hidden = self.setup_decoder(enc_hidden)
        out, dec_hidden, attns = self.decoder(tgt, src,
                                              enc_hidden if dec_hidden is None
                                              else dec_hidden,
                                              context,
                                              init_output)
        return out, attns, dec_hidden
