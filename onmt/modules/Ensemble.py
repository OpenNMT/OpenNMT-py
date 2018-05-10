"""
Ensemble decoding.

Decodes using multiple models simultaneously,
combining their prediction distributions by averaging.
All models in the ensemble must share a target vocabulary.
"""

import torch
import torch.nn as nn

from onmt.Models import DecoderState

class EnsembleDecoderState(DecoderState):
    """Dummy DecoderState that wraps a tuple of real DecoderStates"""
    def __init__(self, model_decoder_states):
        self.model_decoder_states = tuple(model_decoder_states)

    def beam_update(self, idx, positions, beam_size):
        for model_state in self.model_decoder_states:
            model_state.beam_update(idx, positions, beam_size)

    def repeat_beam_size_times(self, beam_size):
        for model_state in self.model_decoder_states:
            model_state.repeat_beam_size_times(beam_size)

class EnsembleDecoder(nn.Module):
    """Dummy Decoder that delegates to individual real Decoders"""
    pass

class EnsembleGenerator(nn.Module):
    """Dummy Generator that delegates to individual real Generators,
    and then averages the resulting target distributions.
    """
    pass
