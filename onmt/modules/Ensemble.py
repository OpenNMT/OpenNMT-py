"""
Ensemble decoding.

Decodes using multiple models simultaneously,
combining their prediction distributions by averaging.
All models in the ensemble must share a target vocabulary.
"""

import torch
import torch.nn as nn

from onmt.Models import DecoderState, EncoderBase, NMTModel
import onmt.ModelConstructor


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


class EnsembleDecoderOutput(object):
    """ Wrapper around multiple decoder final hidden states """
    def __init__(self, model_outputs):
        self.model_outputs = tuple(model_outputs)

    def squeeze(self, dim=None):
        """
        Delegate squeeze to avoid modifying
        :obj:`Translator.translate_batch()`
        """
        return EnsembleDecoderOutput([
            x.squeeze(dim) for x in self.model_outputs])

    def __getitem__(self, index):
        return self.model_outputs[index]


class EnsembleEncoder(EncoderBase):
    """ Dummy Encoder that delegates to individual real Encoders """
    def __init__(self, model_encoders):
        super(EnsembleEncoder, self).__init__()
        self.model_encoders = nn.ModuleList(list(model_encoders))

    def forward(self, src, lengths=None, encoder_state=None):
        enc_hidden, memory_bank = zip(*[
            model_encoder.forward(src, lengths, encoder_state)
            for model_encoder in self.model_encoders])
        return enc_hidden, memory_bank


class EnsembleDecoder(nn.Module):
    """ Dummy Decoder that delegates to individual real Decoders """
    def __init__(self, model_decoders):
        super(EnsembleDecoder, self).__init__()
        self.model_decoders = nn.ModuleList(list(model_decoders))

    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        """ See :obj:`RNNDecoderBase.forward()` """
        # Memory_lengths is a single tensor shared between all models.
        # This assumption will not hold if Translator is modified
        # to calculate memory_lengths as something other than the length
        # of the input.
        outputs, states, attns = zip(*[
            model_decoder.forward(
                tgt, memory_bank[i], state[i], memory_lengths)
            for (i, model_decoder)
            in enumerate(self.model_decoders)])
        mean_attns = self.combine_attns(attns)
        return (EnsembleDecoderOutput(outputs),
                EnsembleDecoderState(states),
                mean_attns)

    def combine_attns(self, attns):
        result = {}
        for key in attns[0].keys():
            result[key] = torch.stack([attn[key] for attn in attns]).mean(0)
        return result

    def init_decoder_state(self, src, memory_bank, enc_hidden):
        """ See :obj:`RNNDecoderBase.init_decoder_state()` """
        return EnsembleDecoderState(
            [model_decoder.init_decoder_state(src,
                                              memory_bank[i],
                                              enc_hidden[i])
             for (i, model_decoder) in enumerate(self.model_decoders)])


class EnsembleGenerator(nn.Module):
    """
    Dummy Generator that delegates to individual real Generators,
    and then averages the resulting target distributions.
    """
    def __init__(self, model_generators):
        self.model_generators = tuple(model_generators)

    def forward(self, hidden):
        """
        Compute a distribution over the target dictionary
        by averaging distributions from models in the ensemble.
        All models in the ensemble must share a target vocabulary.
        """
        distributions = [model_generator.forward(hidden[i])
                         for (i, model_generator)
                         in enumerate(self.model_generators)]
        return torch.stack(distributions).mean(0)


class EnsembleModel(NMTModel):
    """ Dummy NMTModel wrapping individual real NMTModels """
    def __init__(self, models):
        encoder = EnsembleEncoder(model.encoder for model in models)
        decoder = EnsembleDecoder(model.decoder for model in models)
        super(EnsembleModel, self).__init__(encoder, decoder)
        self.generator = EnsembleGenerator(model.generator for model in models)
        self.models = nn.ModuleList(models)


def load_test_model(opt, dummy_opt):
    """ Read in multiple models for ensemble """
    shared_fields = None
    shared_model_opt = None
    models = []
    for model_path in opt.models:
        fields, model, model_opt = \
            onmt.ModelConstructor.load_test_model(opt,
                                                  dummy_opt,
                                                  model_path=model_path)
        if shared_fields is None:
            shared_fields = fields
        else:
            for key, field in fields.items():
                if field is not None and 'vocab' in field.__dict__:
                    assert field.vocab.stoi == shared_fields[key].vocab.stoi, \
                        'Ensemble models must use the same preprocessed data'
        models.append(model)
        if shared_model_opt is None:
            shared_model_opt = model_opt
        # FIXME: anything to check or copy from other model_opt?
    ensemble_model = EnsembleModel(models)
    return shared_fields, ensemble_model, shared_model_opt
