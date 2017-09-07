import torch
import torch.nn as nn
import onmt.modules.GlobalAttention
import line_profiler


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """
    def __init__(self, num_layers, input_size, rnn_size, dropout,
                 multi_attn=False, attn_use_emb=False):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.multi_attn = multi_attn

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

        if self.multi_attn:
            self.attns = nn.ModuleList()
            for i in range(num_layers):
                self.attns.append(onmt.modules.GlobalAttention(
                    rnn_size, use_emb=attn_use_emb
                ))

    def forward(self, input, hidden, context=None, emb=None):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            if self.multi_attn and context is not None:
                attn_output, _ = self.attns[i](h_1_i,
                                               context.transpose(0, 1),
                                               emb=emb.transpose(0, 1)
                                                if emb is not None else None)
                h_1_i = h_1_i + attn_output
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout,
                 multi_attn=False, attn_use_emb=False):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.multi_attn = multi_attn

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

        if self.multi_attn:
            self.attns = nn.ModuleList()
            for i in range(num_layers):
                self.attns.append(onmt.modules.GlobalAttention(
                    rnn_size, use_emb=attn_use_emb
                ))

    def forward(self, input, hidden, context=None, emb=None):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[0][i])
            if self.multi_attn and context is not None:
                attn_output, _ = self.attns[i](h_1_i,
                                               context.transpose(0, 1),
                                               emb.transpose(0, 1)
                                                if emb is not None else None)
                h_1_i = h_1_i + attn_output
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)
        return input, (h_1,)
