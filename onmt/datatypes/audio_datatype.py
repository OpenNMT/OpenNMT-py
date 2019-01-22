# -*- coding: utf-8 -*-
import torch
from torchtext.data import Field

from onmt.datatypes.datatype_base import Datatype
from onmt.inputters.audio_dataset import AudioDataReader


def audio_sort_key(ex):
    """ Sort using duration time of the sound spectrogram. """
    return ex.src.size(1)


def batch_audio(data, vocab):
    """ batch audio data """
    nfft = data[0].size(0)
    t = max([t.size(1) for t in data])
    sounds = torch.zeros(len(data), 1, nfft, t)
    for i, spect in enumerate(data):
        sounds[i, :, :, 0:spect.size(1)] = spect
    return sounds


def audio_fields(base_name, **kwargs):
    audio = Field(
        use_vocab=False, dtype=torch.float,
        postprocessing=batch_audio, sequential=False)

    length = Field(use_vocab=False, dtype=torch.long, sequential=False)

    return [(base_name + "_lengths", length)], [(base_name, audio)]


audio = Datatype("audio", AudioDataReader, audio_sort_key, audio_fields)
