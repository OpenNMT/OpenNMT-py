"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import onmt
from onmt.decoders.decoder import DecoderState
from onmt.utils.misc import aeq

MAX_SIZE = 5000


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            size (int): the size of input for the first-layer of the FFN.
            hidden_size (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """
    def __init__(self, size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.layer_norm = onmt.modules.LayerNorm(size)
        # Save a little memory, by doing inplace.
        self.dropout_1 = nn.Dropout(dropout, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      droput(float): dropout probability(0-1.0).
      head_count(int): the number of heads for MultiHeadedAttention.
      hidden_size(int): the second-layer of the PositionwiseFeedForward.
    """
    def __init__(self, size, dropout,
                 head_count=8, hidden_size=2048):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = onmt.modules.MultiHeadedAttention(
            head_count, size, dropout=dropout)
        self.context_attn = onmt.modules.MultiHeadedAttention(
            head_count, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size,
                                                    hidden_size,
                                                    dropout)
        self.layer_norm_1 = onmt.modules.LayerNorm(size)
        self.layer_norm_2 = onmt.modules.LayerNorm(size)
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                previous_input=None):
        # Args Checks
        input_batch, input_len, _ = inputs.size()
        if previous_input is not None:
            pi_batch, _, _ = previous_input.size()
            aeq(pi_batch, input_batch)
        contxt_batch, contxt_len, _ = memory_bank.size()
        aeq(input_batch, contxt_batch)

        src_batch, _, s_len = src_pad_mask.size()
        tgt_batch, _, _ = tgt_pad_mask.size()
        #src_batch, t_len, s_len = src_pad_mask.size()
        #tgt_batch, t_len_, t_len__ = tgt_pad_mask.size()
        aeq(input_batch, contxt_batch, src_batch, tgt_batch)
        # aeq(t_len, t_len_, t_len__, input_len)
        aeq(s_len, contxt_len)
        # END Args Checks

        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None
        query, attn = self.self_attn(all_input, all_input, input_norm,
                                     mask=dec_mask)
        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask)
        output = self.feed_forward(self.drop(mid) + query)

        # CHECKS
        output_batch, output_len, _ = output.size()
        aeq(input_len, output_len)
        aeq(contxt_batch, output_batch)

        n_batch_, t_len_, s_len_ = attn.size()
        aeq(input_batch, n_batch_)
        aeq(contxt_len, s_len_)
        aeq(input_len, t_len_)
        # END CHECKS

        return output, attn, all_input

    def _get_attn_subsequent_mask(self, size):
        ''' Get an attention mask to avoid using the subsequent info.'''
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       hidden_size (int): number of hidden units
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

       attn_type (str): if using a seperate copy attention
    """
    def __init__(self, num_layers, hidden_size, attn_type,
                 copy_attn, dropout, embeddings):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(hidden_size, dropout)
             for _ in range(num_layers)])

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type)
            self._copy = True
        self.layer_norm = onmt.modules.LayerNorm(hidden_size)

    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        # CHECKS
        assert isinstance(state, TransformerDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        memory_len, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)

        src = state.src
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()
        aeq(tgt_batch, memory_batch, src_batch, tgt_batch)
        aeq(memory_len, src_len)

        if state.previous_input is not None:
            tgt = torch.cat([state.previous_input, tgt], 0)
        # END CHECKS

        # Initialize return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt)
        if state.previous_input is not None:
            emb = emb[state.previous_input.size(0):, ]
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, tgt_len, src_len)
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        saved_inputs = []
        for i in range(self.num_layers):
            prev_layer_input = None
            if state.previous_input is not None:
                prev_layer_input = state.previous_layer_inputs[i]
            output, attn, all_input \
                = self.transformer_layers[i](output, src_memory_bank,
                                             src_pad_mask, tgt_pad_mask,
                                             previous_input=prev_layer_input)
            saved_inputs.append(all_input)

        saved_inputs = torch.stack(saved_inputs)
        output = self.layer_norm(output)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns["std"] = attn
        if self._copy:
            attns["copy"] = attn

        # Update the state.
        state = state.update_state(tgt, saved_inputs)
        return outputs, state, attns

    def init_decoder_state(self, src, memory_bank, enc_hidden):
        """ Init decoder state """
        return TransformerDecoderState(src)


class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """
    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        return (self.previous_input, self.previous_layer_inputs, self.src)

    def update_state(self, new_input, previous_layer_inputs):
        """ Called for every decoder forward pass. """
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = Variable(self.src.data.repeat(1, beam_size, 1),
                            volatile=True)
