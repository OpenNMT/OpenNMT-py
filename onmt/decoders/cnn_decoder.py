"""
Implementation of the CNN Decoder part of
  "Convolutional Sequence to Sequence Learning"
"""
import torch
import torch.nn as nn

import onmt.modules
from onmt.decoders.decoder import DecoderState
from onmt.utils.misc import aeq
from onmt.utils.cnn_factory import shape_transform, GatedConv

SCALE_WEIGHT = 0.5 ** 0.5


class CNNDecoder(nn.Module):
    """
    Decoder built on CNN, based on :cite:`DBLP:journals/corr/GehringAGYD17`.


    Consists of residual convolutional layers, with ConvMultiStepAttention.
    """

    def __init__(self, num_layers, hidden_size, attn_type,
                 copy_attn, cnn_kernel_width, dropout, embeddings):
        super(CNNDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'cnn'
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cnn_kernel_width = cnn_kernel_width
        self.embeddings = embeddings
        self.dropout = dropout

        # Build the CNN.
        input_size = self.embeddings.embedding_size
        self.linear = nn.Linear(input_size, self.hidden_size)
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.conv_layers.append(
                GatedConv(self.hidden_size, self.cnn_kernel_width,
                          self.dropout, True))

        self.attn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.attn_layers.append(
                onmt.modules.ConvMultiStepAttention(self.hidden_size))

        # CNNDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type)
            self._copy = True

    def forward(self, tgt, memory_bank, state, memory_lengths=None, step=None):
        """ See :obj:`onmt.modules.RNNDecoderBase.forward()`"""
        # NOTE: memory_lengths is only here for compatibility reasons
        #       with onmt.modules.RNNDecoderBase.forward()
        # CHECKS
        assert isinstance(state, CNNDecoderState)
        _, tgt_batch, _ = tgt.size()
        _, contxt_batch, _ = memory_bank.size()
        aeq(tgt_batch, contxt_batch)
        # END CHECKS

        if state.previous_input is not None:
            tgt = torch.cat([state.previous_input, tgt], 0)

        # Initialize return variables.
        outputs = []
        attns = {"std": []}
        assert not self._copy, "Copy mechanism not yet tested in conv2conv"
        if self._copy:
            attns["copy"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        tgt_emb = emb.transpose(0, 1).contiguous()
        # The output of CNNEncoder.
        src_memory_bank_t = memory_bank.transpose(0, 1).contiguous()
        # The combination of output of CNNEncoder and source embeddings.
        src_memory_bank_c = state.init_src.transpose(0, 1).contiguous()

        # Run the forward pass of the CNNDecoder.
        emb_reshape = tgt_emb.contiguous().view(
            tgt_emb.size(0) * tgt_emb.size(1), -1)
        linear_out = self.linear(emb_reshape)
        x = linear_out.view(tgt_emb.size(0), tgt_emb.size(1), -1)
        x = shape_transform(x)

        pad = torch.zeros(x.size(0), x.size(1),
                          self.cnn_kernel_width - 1, 1)

        pad = pad.type_as(x)
        base_target_emb = x

        for conv, attention in zip(self.conv_layers, self.attn_layers):
            new_target_input = torch.cat([pad, x], 2)
            out = conv(new_target_input)
            c, attn = attention(base_target_emb, out,
                                src_memory_bank_t, src_memory_bank_c)
            x = (x + (c + out) * SCALE_WEIGHT) * SCALE_WEIGHT
        output = x.squeeze(3).transpose(1, 2)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()
        if state.previous_input is not None:
            outputs = outputs[state.previous_input.size(0):]
            attn = attn[:, state.previous_input.size(0):].squeeze()
            attn = torch.stack([attn])
        attns["std"] = attn
        if self._copy:
            attns["copy"] = attn

        # Update the state.
        state.update_state(tgt)

        return outputs, state, attns

    def init_decoder_state(self, _, memory_bank, enc_hidden, with_cache=False):
        """
        Init decoder state.
        """
        return CNNDecoderState(memory_bank, enc_hidden)


class CNNDecoderState(DecoderState):
    """
    Init CNN decoder state.
    """

    def __init__(self, memory_bank, enc_hidden):
        self.init_src = (memory_bank + enc_hidden) * SCALE_WEIGHT
        self.previous_input = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        return (self.previous_input,)

    def detach(self):
        self.previous_input = self.previous_input.detach()

    def update_state(self, new_input):
        """ Called for every decoder forward pass. """
        self.previous_input = new_input

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.init_src = self.init_src.data.repeat(1, beam_size, 1)
