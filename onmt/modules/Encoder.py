import torch.nn as nn
import onmt
import onmt.modules
from onmt.Utils import aeq
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class MeanEncoder(nn.Module):
    """
    Mean encoder: a small module for computing the mean of a sequence of
    embeddings.
    """
    def __init__(self, num_layers):
        self.layers = num_layers
        super(MeanEncoder, self).__init__()

    def forward(self, emb, **kwargs):
        """
        emb (FloatTensor): src_len x batch x embedding dimension
        returns:
        """
        _, n_batch, emb_dim = emb.size()
        mean = emb.mean(0).expand(self.layers, n_batch, emb_dim)
        return (mean, mean), emb


class RNNEncoder(nn.Module):
    """
    Standard RNN encoder. Handles layer sizing issues for bidirectional
    networks and allows for sequence packing.
    """
    def __init__(self, rnn_type, embedding_dim, rnn_size,
                 num_layers, dropout, bidirectional):
        assert rnn_type in ['LSTM', 'GRU']
        # TODO: implement sum BRNN merge
        num_directions = 2 if bidirectional else 1
        assert rnn_size % num_directions == 0
        hidden_size = rnn_size // num_directions
        super(RNNEncoder, self).__init__()
        self.rnn = getattr(nn, rnn_type)(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional)

    def forward(self, emb, lengths=None, hidden=None, **kwargs):
        packed_emb = emb
        if lengths is not None:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)
        outputs, hidden_t = self.rnn(packed_emb, hidden)
        if lengths:
            outputs = unpack(outputs)[0]
        return hidden_t, outputs


class Encoder(nn.Module):
    """
    Encoder recurrent neural network. The encoder consists of two basic
    parts: an embedding matrix for mapping input tokens to fixed-dimensional
    vectors, and a unit which processes these embedded tokens.
    """

    def __init__(self, encoder_type, bidirectional, rnn_type,
                 num_layers, rnn_size, dropout, embeddings,
                 cnn_kernel_width=3):
        """
        Args:
            encoder_type (string): rnn, brnn, mean, or transformer.
            bidirectional (bool): bidirectional Encoder.
            rnn_type (string): LSTM or GRU.
            num_layers (int): number of Encoder layers.
            rnn_size (int): size of hidden states of a rnn.
            dropout (float): dropout probablity.
            embeddings (Embeddings): vocab embeddings for this Encoder.
        """
        super(Encoder, self).__init__()

        self.embeddings = embeddings

        # Build the Encoder RNN.
        if encoder_type == "mean":
            self.encoder = MeanEncoder(num_layers)
        elif encoder_type == "transformer":
            self.encoder = onmt.modules.TransformerEncoder(
                rnn_size, dropout, num_layers,
                self.embeddings.word_padding_idx)
        elif encoder_type == "cnn":
            self.encoder = onmt.modules.CNNEncoder(
                self.embeddings.embedding_size, rnn_size,
                dropout, num_layers, cnn_kernel_width)
        else:
            self.encoder = RNNEncoder(
                rnn_type, self.embeddings.embedding_size, rnn_size,
                num_layers, dropout, bidirectional)

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
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

        emb = self.embeddings(input)

        return self.encoder(emb, input=input, lengths=lengths, hidden=hidden)
