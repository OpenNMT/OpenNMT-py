import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import math
import numpy as np
import time

def make_positional_encodings(dim, max_len):
    pe = torch.FloatTensor(max_len, 1, dim).fill_(0)
    print(pe.size())
    for i in range(dim):
        for j in range(max_len):
            k = float(j) / (10000.0 ** (2.0*i / float(dim)))
            pe[j, 0, i] = math.cos(k) if i % 2 == 1 else math.sin(k)
    return pe

def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_k = seq_k.size()
    mb_size, len_q = seq_q.size()
    pad_attn_mask = seq_k.data.eq(onmt.Constants.PAD).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask

def get_attn_subsequent_mask(size):
    ''' Get an attention mask to avoid using the subsequent info.'''
    # assert seq.dim() == 2
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask


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
        self.dropout = nn.Dropout(p=opt.dropout)

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
        self.positional_encoding = opt.position_encoding \
                                   if "position_encoding" in opt else ""
        self.rnn = nn.LSTM(input_size, self.hidden_size,
                           num_layers=opt.layers,
                           dropout=opt.dropout,
                           bidirectional=opt.brnn)
        self.word_vec_size = opt.word_vec_size

        if self.positional_encoding:
            self.pe = make_positional_encodings(opt.word_vec_size, 5000).cuda()
        if self.encoder_layer == "transformer":
            self.transformer = nn.ModuleList([TransformerEncoder(self.hidden_size, opt)
                                               for i in range(opt.layers)])

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
            # emb = emb * math.sqrt(emb.size(2))
            # if self.encoder_layer == "transformer":
            emb = emb + Variable(self.pe[:emb.size(0), :1, :emb.size(2)].expand_as(emb))
            # emb = emb * math.sqrt(self.word_vec_size)
            emb = self.dropout(emb)
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
        if lengths is not None:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.data.view(-1).tolist()
            pre_emb = self._embed(input)
            emb = pack(pre_emb, lengths)
        else:
            emb = self._embed(input)
            pre_emb = emb

        if self.encoder_layer == "mean":
            # Just take the mean of vectors.
            mean = pre_emb.mean(0).view(1, pre_emb.size(1), pre_emb.size(2)) \
                   .expand(self.layers, pre_emb.size(1), pre_emb.size(2))
            return (mean, mean), pre_emb

        if self.encoder_layer == "transformer":
            # Self-attention tranformer.
            mean = pre_emb.mean(0).view(1, pre_emb.size(1), pre_emb.size(2)) \
                   .expand(self.layers, pre_emb.size(1), pre_emb.size(2))
            out = pre_emb.transpose(0, 1).contiguous()
            for i in range(self.layers):
                out = self.transformer[i](out, input[:, :, 0])
            return (mean, mean), out.transpose(0, 1).contiguous()


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


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, opt, use_struct=False):
        super(TransformerEncoder, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(8, hidden_size,
                                                           p=opt.dropout,
                                                           use_struct=use_struct)
        self.feed_forward = onmt.modules.PositionwiseFeedForward(hidden_size, 2048,
                                                                 opt.dropout)

    def forward(self, input, words):
        start = time.time()

        mask = get_attn_padding_mask(words.transpose(0,1), words.transpose(0,1))
        mid, _ = self.self_attn(input, input, input, mask=mask)
        out = self.feed_forward(mid)
        return out

class TransformerDecoder(nn.Module):
    """
    The Transformer Decoder from AIAYN
    """
    def __init__(self, hidden_size, opt):
        super(TransformerDecoder, self).__init__()
        self.self_attn = onmt.modules.MultiHeadedAttention(8, hidden_size,
                                                           p=opt.dropout)
        self.context_attn = onmt.modules.MultiHeadedAttention(8, hidden_size,
                                                              p=opt.dropout)
        self.feed_forward = onmt.modules.PositionwiseFeedForward(hidden_size, 2048,
                                                                 opt.dropout)
        self.dropout = opt.dropout
        self.mask = get_attn_subsequent_mask(5000).cuda()

    def forward(self, input, context, src_words, tgt_words):
        """
        Args:
            input : batch x len x hidden
            context : batch x qlen x hidden
        Returns:
            output : batch x len x hidden
            attn : batch x len x qlen
        """
        start = time.time()
        attn_mask = get_attn_padding_mask(tgt_words.transpose(0,1), tgt_words.transpose(0,1))
        # bxsqxsk
        # sub_mask = get_attn_subsequent_mask(tgt_words.transpose(0,1))
        # bxsqxsq
        dec_mask = torch.gt(attn_mask + self.mask[:, :attn_mask.size(1), :attn_mask.size(1)].expand_as(attn_mask), 0)
        # dec_mask = attn_mask

        pad_mask = get_attn_padding_mask(tgt_words.transpose(0,1), src_words.transpose(0,1))
        # bxsqxsk
        start2 = time.time()
        # def input_hook(grad):
        #     print("input grad", grad[0].sum(1))
        # input.register_hook(input_hook)
        # input = input.fill(0)
        # input[0, 7, :].fill(1)
        # input = Variable(input.data, requires_grad=True)

        query, attn = self.self_attn(input, input, input, mask=dec_mask)
        # print("ATTN", " ".join(["%3f"%j for j in attn.data[0, 10]]))

        # def attn_hook(grad):
        #     print(grad.sum())
        #     grad = grad.view(grad.size(0) // 8, 8, grad.size(1), grad.size(2))
        #     for i in range(8):
        #         print("GRAD",i, " ".join(["%3f"%j for j in grad.data[0, i, 10]]))
        #         print(grad[0, i, 9].sum())
        #     print()

        # def query_hook(grad):
        #     print("Out", "%3f" % grad.data[0, 10].sum())

        # query.register_hook(query_hook)
        # query[0, 10, 0].backward()
        # exit()
        # self.self_attn.register_backward_hook(
        #     lambda _, grad_input, grad_output: print(len(grad_output)))
        mid, attn = self.context_attn(context, context, query, mask=pad_mask)
        # mid = query
        output = self.feed_forward(mid)
        # output[0, 10, 0].backward()
        # exit()
        # ff_out = self.layer2(F.relu(self.layer1(mid.contiguous())))
        # output = self.norm(mid, ff_out)
        # output[3, 7, 0].backward()
        # print(input.grad[3, 1, 0])
        # print(input.grad[3, 6, 0])
        # print(input.grad[3, 7, 0])
        # print(input.grad[3, 8, 0])
        # exit()
        # print("decoder2", time.time() - start2)
        # print("decoder", time.time() - start)
        return output, attn

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
        self.decoder_layer = opt.decoder_layer \
                             if "decoder_layer" in opt else ""
        self.positional_encoding = opt.position_encoding \
                                   if "position_encoding" in opt else ""

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        self.rnn = StackedLSTM(opt.layers, input_size,
                               opt.rnn_size, opt.dropout)
        self.dropout = nn.Dropout(opt.dropout)
        self.emb_dropout = nn.Dropout(opt.dropout)
        self.hidden_size = opt.rnn_size

        self._coverage = opt.coverage_attn if "coverage_attn" in opt else False
        # Std attention layer.
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size, self._coverage)

        # Separate Copy Attention.
        self._copy = False
        if opt.copy_attn if "copy_attn" in opt else False:
            self.copy_attn = onmt.modules.GlobalAttention(opt.rnn_size)
            self._copy = True


        if self.positional_encoding:
            self.pe = make_positional_encodings(opt.word_vec_size, 5000).cuda()

        if self.decoder_layer == "transformer":
            self.transformer = nn.ModuleList([TransformerDecoder(self.hidden_size, opt)
                                               for _ in range(opt.layers)])

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, src, hidden, context, init_feed):
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
        if False:
            if self.decoder_layer == "transformer" and hidden is not None:
                input = torch.cat([hidden, input], 0)

        emb = self.word_lut(input)
        if self.positional_encoding:
            emb = emb + Variable(self.pe[:emb.size(0), :1, :emb.size(2)].expand_as(emb))
            # emb = emb * math.sqrt(emb.size(2))

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        output = init_feed
        coverage = None

        if self.decoder_layer == "transformer":
            output = emb.transpose(0, 1).contiguous()
            output = self.emb_dropout(output)

            src_context = context.transpose(0,1).contiguous()
            for i in range(self.layers):
                output, attn = self.transformer[i](output, src_context, src[:,:,0], input)
            outputs = output.transpose(0, 1).contiguous()

            if False:
                if hidden is not None:
                    outputs = outputs[hidden.size(0):]
                    attn = attn[:, hidden.size(0):].squeeze()
                    attn = torch.stack([attn])

            attns["std"] = attn
            if self._copy:
                attns["copy"] = attn
            hidden = input
        else:
            src_context = context.transpose(0,1).contiguous()
            for i, emb_t in enumerate(emb.split(1)):
                emb_t = emb_t.squeeze(0)
                if self.input_feed:
                    emb_t = torch.cat([emb_t, output], 1)


                output, hidden = self.rnn(emb_t, hidden)
                output, attn = self.attn(output, context.transpose(0, 1), coverage)

                if self._coverage:
                    if coverage:
                        coverage = coverage + attn
                    else:
                        coverage = attn
                    attns["coverage"] += [coverage]

                output = self.dropout(output)
                outputs += [output]
                attns["std"] += [attn]


                # COPY
                if self._copy:
                    _, copy_attn = self.copy_attn(output, context.t())
                    attns["copy"] += [copy_attn]

                # elif self._copy:
                #     attns["copy"] += [attn]
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
        start = time.time()
        src = input.src
        tgt = input.tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, input.lengths)
        init_output = self.make_init_decoder_output(context)

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
        out, dec_hidden, attns = self.decoder(tgt, src,
                                              enc_hidden if dec_hidden is None
                                              else dec_hidden,
                                              context,
                                              init_output)
        # print(time.time() - start)
        return out, attns, dec_hidden
