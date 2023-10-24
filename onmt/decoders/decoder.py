import torch
import torch.nn as nn

from onmt.modules.stacked_rnn import StackedLSTM, StackedGRU
from onmt.modules import context_gate_factory, GlobalAttention
from onmt.utils.rnn_factory import rnn_factory


class DecoderBase(nn.Module):
    """Abstract class for decoders.

    Args:
        attentional (bool): The decoder returns non-empty attention.
    """

    def __init__(self, attentional=True):
        super(DecoderBase, self).__init__()
        self.attentional = attentional

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor.

        Subclasses should override this method.
        """

        raise NotImplementedError


class RNNDecoderBase(DecoderBase):
    """Base recurrent attention-based decoder class.

    Specifies the interface used by different decoder types
    and required by :class:`~onmt.models.NMTModel`.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :class:`~onmt.modules.GlobalAttention`
       attn_func (str) : see :class:`~onmt.modules.GlobalAttention`
       coverage_attn (str): see :class:`~onmt.modules.GlobalAttention`
       context_gate (str): see :class:`~onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
       reuse_copy_attn (bool): reuse the attention for copying
       copy_attn_type (str): The copy attention style. See
        :class:`~onmt.modules.GlobalAttention`.
    """

    def __init__(
        self,
        rnn_type,
        bidirectional_encoder,
        num_layers,
        hidden_size,
        attn_type="general",
        attn_func="softmax",
        coverage_attn=False,
        context_gate=None,
        copy_attn=False,
        dropout=0.0,
        embeddings=None,
        reuse_copy_attn=False,
        copy_attn_type="general",
    ):
        super(RNNDecoderBase, self).__init__(
            attentional=attn_type != "none" and attn_type is not None
        )

        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn = self._build_rnn(
            rnn_type,
            input_size=self._input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = context_gate_factory(
                context_gate, self._input_size, hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        if not self.attentional:
            if self._coverage:
                raise ValueError("Cannot use coverage term with no attention.")
            self.attn = None
        else:
            self.attn = GlobalAttention(
                hidden_size,
                coverage=coverage_attn,
                attn_type=attn_type,
                attn_func=attn_func,
            )

        if copy_attn and not reuse_copy_attn:
            if copy_attn_type == "none" or copy_attn_type is None:
                raise ValueError("Cannot use copy_attn with copy_attn_type none")
            self.copy_attn = GlobalAttention(
                hidden_size, attn_type=copy_attn_type, attn_func=attn_func
            )
        else:
            self.copy_attn = None

        self._reuse_copy_attn = reuse_copy_attn and copy_attn
        if self._reuse_copy_attn and not self.attentional:
            raise ValueError("Cannot reuse copy attention with no attention.")

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.dec_layers,
            opt.dec_hid_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.coverage_attn,
            opt.context_gate,
            opt.copy_attn,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.reuse_copy_attn,
            opt.copy_attn_type,
        )

    def init_state(self, src, _, enc_final_hs):
        """Initialize decoder state with last state of the encoder."""

        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat(
                    [hidden[0 : hidden.size(0) : 2], hidden[1 : hidden.size(0) : 2]], 2
                )
            return hidden

        if isinstance(enc_final_hs, tuple):  # LSTM
            self.state["hidden"] = tuple(
                _fix_enc_hidden(enc_hid) for enc_hid in enc_final_hs
            )
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(enc_final_hs),)

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)

        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = (
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        )

        self.state["coverage"] = None

    def map_state(self, fn):
        self.state["hidden"] = tuple(
            fn(h.transpose(0, 1), 0).transpose(0, 1) for h in self.state["hidden"]
        )
        self.state["input_feed"] = fn(
            self.state["input_feed"].transpose(0, 1), 0
        ).transpose(0, 1)
        if self._coverage and self.state["coverage"] is not None:
            self.state["coverage"] = fn(
                self.state["coverage"].transpose(0, 1), 0
            ).transpose(0, 1)

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.state["input_feed"] = self.state["input_feed"].detach()
        if self._coverage and self.state["coverage"] is not None:
            self.state["coverage"] = self.state["coverage"].detach()

    def forward(self, tgt, enc_out, src_len=None, step=None, **kwargs):
        """
        Args:
            tgt (LongTensor): sequences of padded tokens
                 ``(batch, tgt_len, nfeats)``.
            enc_out (FloatTensor): vectors from the encoder
                 ``(batch, src_len, hidden)``.
            src_len (LongTensor): the padded source lengths
                ``(batch,)``.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(batch, tgt_len, hidden)``.
            * attns: distribution over src at each tgt
              ``(batch, tgt_len, src_len)``.
        """
        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, enc_out, src_len=src_len
        )

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs, dim=1)
            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])

        self.state["input_feed"] = dec_outs[:, -1, :].unsqueeze(0)
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1, :, :].unsqueeze(0)

        return dec_outs, attns

    def update_dropout(self, dropout, attention_dropout=None):
        self.dropout.p = dropout
        self.embeddings.update_dropout(dropout)


class StdRNNDecoder(RNNDecoderBase):
    """Standard fully batched RNN decoder with attention.

    Faster implementation, uses CuDNN for implementation.
    See :class:`~onmt.decoders.decoder.RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """

    def _run_forward_pass(self, tgt, enc_out, src_len=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.

        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                ``(batch, tgt_len, nfeats)``.
            enc_out (FloatTensor): output(tensor sequence) from the
                encoder RNN of size ``(batch, src_len, hidden_size)``.
            src_len (LongTensor): the source enc_out lengths.

        Returns:
            (Tensor, List[FloatTensor], Dict[str, List[FloatTensor]):

            * dec_state: final hidden state from the decoder.
            * dec_outs: an array of output of every time
              step from the decoder.
            * attns: a dictionary of different
              type of attention Tensor array of every time
              step from the decoder.
        """

        assert self.copy_attn is None  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        attns = {}
        emb = self.embeddings(tgt)

        if isinstance(self.rnn, nn.GRU):
            rnn_out, dec_state = self.rnn(emb, self.state["hidden"][0])
        else:
            rnn_out, dec_state = self.rnn(emb, self.state["hidden"])

        tgt_batch, tgt_len, _ = tgt.size()

        # Calculate the attention.
        if not self.attentional:
            dec_outs = rnn_out
        else:
            dec_outs, p_attn = self.attn(rnn_out, enc_out, src_len=src_len)
            attns["std"] = p_attn

        # Calculate the context gate.
        if self.context_gate is not None:
            dec_outs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_out.view(-1, rnn_out.size(2)),
                dec_outs.view(-1, dec_outs.size(2)),
            )
            dec_outs = dec_outs.view(tgt_batch, tgt_len, self.hidden_size)

        dec_outs = self.dropout(dec_outs)

        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """Input feeding based decoder.

    See :class:`~onmt.decoders.decoder.RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`

    """

    def _run_forward_pass(self, tgt, enc_out, src_len=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)

        dec_outs = []
        attns = {}
        if self.attn is not None:
            attns["std"] = []
        if self.copy_attn is not None or self._reuse_copy_attn:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # batch x len x embedding_dim

        dec_state = self.state["hidden"]

        coverage = (
            self.state["coverage"].squeeze(0)
            if self.state["coverage"] is not None
            else None
        )

        # Input feed concatenates hidden state with
        # input at every time step.
        for emb_t in emb.split(1, dim=1):
            dec_in = torch.cat([emb_t.squeeze(1), input_feed], 1)
            rnn_out, dec_state = self.rnn(dec_in, dec_state)
            if self.attentional:
                dec_out, p_attn = self.attn(rnn_out, enc_out, src_len=src_len)
                attns["std"].append(p_attn)
            else:
                dec_out = rnn_out
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                dec_out = self.context_gate(dec_in, rnn_out, dec_out)
            dec_out = self.dropout(dec_out)
            input_feed = dec_out

            dec_outs += [dec_out]

            # Update the coverage attention.
            # attns["coverage"] is actually c^(t+1) of See et al(2017)
            # 1-index shifted
            if self._coverage:
                coverage = p_attn if coverage is None else p_attn + coverage
                attns["coverage"] += [coverage]

            if self.copy_attn is not None:
                _, copy_attn = self.copy_attn(dec_out, enc_out)
                attns["copy"] += [copy_attn]
            elif self._reuse_copy_attn:
                attns["copy"] = attns["std"]

        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        assert rnn_type != "SRU", (
            "SRU doesn't support input feed! " "Please set -input_feed 0!"
        )
        stacked_cell = StackedLSTM if rnn_type == "LSTM" else StackedGRU
        return stacked_cell(num_layers, input_size, hidden_size, dropout)

    @property
    def _input_size(self):
        """Using input feed by concatenating input with attention vectors."""
        return self.embeddings.embedding_size + self.hidden_size

    def update_dropout(self, dropout, attention_dropout=None):
        self.dropout.p = dropout
        self.rnn.dropout.p = dropout
        self.embeddings.update_dropout(dropout)
