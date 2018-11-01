""" Base Class and function for Decoders """

from __future__ import division
import torch
import torch.nn as nn

import onmt.models.stacked_rnn
from onmt.utils.misc import aeq
from onmt.utils.rnn_factory import rnn_factory


class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[memory_bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general", attn_func="softmax",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type, attn_func=attn_func
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type, attn_func=attn_func
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def init_state(self, src, memory_bank, encoder_final, with_cache=False):
        """ Init decoder state with last state of the encoder """
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple([_fix_enc_hidden(enc_hid)
                                          for enc_hid in encoder_final])
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final), )

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["coverage"] = None

    def update_state(self, rnnstate, input_feed, coverage):
        """ Update decoder state """
        if not isinstance(rnnstate, tuple):
            self.state["hidden"] = (rnnstate,)
        else:
            self.state["hidden"] = rnnstate
        self.state["input_feed"] = input_feed
        self.state["coverage"] = coverage

    def map_state(self, fn):
        self.state["hidden"] = tuple(map(lambda x: fn(x, 1),
                                         self.state["hidden"]))
        self.state["input_feed"] = fn(self.state["input_feed"], 1)

    def detach_state(self):
        """ Need to document this """
        self.state["hidden"] = tuple([_.detach()
                                     for _ in self.state["hidden"]])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self, tgt, memory_bank, memory_lengths=None,
                step=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * dec_outs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Run the forward pass of the RNN.
        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, memory_bank, memory_lengths=memory_lengths)

        # Update the state with the result.
        output = dec_outs[-1]
        coverage = None
        if "coverage" in attns:
            coverage = attns["coverage"][-1].unsqueeze(0)
        self.update_state(dec_state, output.unsqueeze(0), coverage)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns


class StdRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                 [len x batch x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the
                          encoder RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            dec_state (Tensor): final hidden state from the decoder.
            dec_outs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        attns = {}
        emb = self.embeddings(tgt)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, dec_state = self.rnn(emb, self.state["hidden"][0])
        else:
            rnn_output, dec_state = self.rnn(emb, self.state["hidden"])

        # Check
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        # END

        # Calculate the attention.
        dec_outs, p_attn = self.attn(
            rnn_output.transpose(0, 1).contiguous(),
            memory_bank.transpose(0, 1),
            memory_lengths=memory_lengths
        )
        attns["std"] = p_attn

        # Calculate the context gate.
        if self.context_gate is not None:
            dec_outs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                dec_outs.view(-1, dec_outs.size(2))
            )
            dec_outs = \
                dec_outs.view(tgt_len, tgt_batch, self.hidden_size)

        dec_outs = self.dropout(dec_outs)
        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[memory_bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # Initialize local and return variables.
        dec_outs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]
        coverage = self.state["coverage"].squeeze(0) \
            if self.state["coverage"] is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for _, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)
            decoder_output, p_attn = self.attn(
                rnn_output,
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths)
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            dec_outs += [decoder_output]
            attns["std"] += [p_attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output,
                                              memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
        # Return result.
        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.models.stacked_rnn.StackedLSTM
        else:
            stacked_cell = onmt.models.stacked_rnn.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size
