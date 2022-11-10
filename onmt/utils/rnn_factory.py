"""
 RNN tools
"""
import torch.nn as nn
import onmt.models


def rnn_factory(rnn_type, **kwargs):
    """ rnn factory, Use pytorch version when available. """
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        # SRU doesn't support PackedSequence.
        no_pack_padded_seq = True
        rnn = onmt.models.sru.SRU(batch_first=True, **kwargs)
    else:
        rnn = getattr(nn, rnn_type)(batch_first=True, **kwargs)
    return rnn, no_pack_padded_seq
