"""
Ensemble decoding.

Decodes using multiple models simultaneously,
combining their prediction distributions by averaging.
All models in the ensemble must share a target vocabulary.
"""

import torch
import torch.nn as nn

from onmt.Models import DecoderState, EncoderBase


class EnsembleDecoderState(DecoderState):
    """ Dummy DecoderState that wraps a tuple of real DecoderStates """
    def __init__(self, model_decoder_states):
        self.model_decoder_states = tuple(model_decoder_states)

    def beam_update(self, idx, positions, beam_size):
        for model_state in self.model_decoder_states:
            model_state.beam_update(idx, positions, beam_size)

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        for model_state in self.model_decoder_states:
            model_state.repeat_beam_size_times(beam_size)

    def __getitem__(self, index):
        return self.model_decoder_states[index]


class EnsembleDistribution(object):
    """ Wrapper around multiple prediction distributions """
    def __init__(self, model_outputs):
        self.model_outputs = tuple(model_outputs)

    def squeeze(self, dim=None):
        return EnsembleDistribution([
            x.squeeze(dim) for x in self.model_outputs])


class EnsembleEncoder(EncoderBase):
    """ Dummy Encoder that delegates to individual real Encoders """
    def __init__(self, model_encoders):
        self.model_encoders = tuple(model_encoders)

    def forward(self, src, lengths=None, encoder_state=None):
        enc_hidden, memory_bank = zip(*[
            model_encoder.forward(src, lengths, encoder_state)
            for model_encoder in self.model_encoders])
        return enc_hidden, memory_bank


class EnsembleDecoder(nn.Module):
    """ Dummy Decoder that delegates to individual real Decoders """
    def __init__(self, model_decoders):
        self.model_decoders = tuple(model_decoders)

    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        """ See :obj:`RNNDecoderBase.forward()` """
        if memory_lengths is None:
            memory_lengths = [None] * len(self.model_decoders)
        outputs, states, attns = zip(*[
            model_decoder.forward(
                tgt, memory_bank[i], state[i], memory_lengths[i])
            for (i, model_decoder)
            in enumerate(self.model_decoders)])
        return (EnsembleDistribution(outputs),
                EnsembleDecoderState(states),
                torch.stack(attns).mean(0))

    def init_decoder_state(self, src, memory_bank, enc_hidden):
        return EnsembleDecoderState(
            [model_decoder.init_decoder_state(src, memory_bank, enc_hidden)
             for model_decoder in self.model_decoders])

    def get_memory_lengths(self, memory_banks):
        """ Get sequence lengths from each memory_bank """
        return [model_decoder.get_memory_lengths(memory_bank)
                for (memory_bank, model_decoder)
                in zip(memory_banks, self.model_decoders)]


class EnsembleGenerator(nn.Module):
    """
    Dummy Generator that delegates to individual real Generators,
    and then averages the resulting target distributions.
    """
    pass
