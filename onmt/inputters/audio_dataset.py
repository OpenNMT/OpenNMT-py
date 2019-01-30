# -*- coding: utf-8 -*-
import os
from tqdm import tqdm

import torch
from torchtext.data import Field

from onmt.inputters.dataset_base import DatasetBase

# imports of datatype-specific dependencies
try:
    import torchaudio
    import librosa
    import numpy as np
except ImportError:
    torchaudio, librosa, np = None, None, None


class AudioDataset(DatasetBase):
    @staticmethod
    def _check_deps():
        if any([torchaudio is None, librosa is None, np is None]):
            AudioDataset._raise_missing_dep(
                "torchaudio", "librosa", "numpy")

    @staticmethod
    def sort_key(ex):
        """ Sort using duration time of the sound spectrogram. """
        return ex.src.size(1)

    @staticmethod
    def extract_features(audio_path, sample_rate, truncate, window_size,
                         window_stride, window, normalize_audio):
        # torchaudio loading options recently changed. It's probably
        # straightforward to rewrite the audio handling to make use of
        # up-to-date torchaudio, but in the meantime there is a legacy
        # method which uses the old defaults
        AudioDataset._check_deps()
        sound, sample_rate_ = torchaudio.legacy.load(audio_path)
        if truncate and truncate > 0:
            if sound.size(0) > truncate:
                sound = sound[:truncate]

        assert sample_rate_ == sample_rate, \
            'Sample rate of %s != -sample_rate (%d vs %d)' \
            % (audio_path, sample_rate_, sample_rate)

        sound = sound.numpy()
        if len(sound.shape) > 1:
            if sound.shape[1] == 1:
                sound = sound.squeeze()
            else:
                sound = sound.mean(axis=1)  # average multiple channels

        n_fft = int(sample_rate * window_size)
        win_length = n_fft
        hop_length = int(sample_rate * window_stride)
        # STFT
        d = librosa.stft(sound, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=window)
        spect, _ = librosa.magphase(d)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if normalize_audio:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
        return spect

    @classmethod
    def make_examples(
        cls,
        data,
        src_dir,
        side,
        sample_rate,
        window_size,
        window_stride,
        window,
        normalize_audio,
        truncate=None
    ):
        """
        Args:
            data: sequence of audio paths or path containing these sequences
            src_dir (str): location of source audio files.
            side (str): 'src' or 'tgt'.
            sample_rate (int): sample_rate.
            window_size (float) : window size for spectrogram in seconds.
            window_stride (float): window stride for spectrogram in seconds.
            window (str): window type for spectrogram generation.
            normalize_audio (bool): subtract spectrogram by mean and divide
                by std or not.
            truncate (int): maximum audio length (0 or None for unlimited).

        Yields:
            a dictionary containing audio data for each line.
        """
        assert src_dir is not None and os.path.exists(src_dir),\
            "src_dir must be a valid directory if data_type is audio"

        if isinstance(data, str):
            data = cls._read_file(data)

        for i, line in enumerate(tqdm(data)):
            line = line.decode("utf-8").strip()
            audio_path = os.path.join(src_dir, line)
            if not os.path.exists(audio_path):
                audio_path = line

            assert os.path.exists(audio_path), \
                'audio path %s not found' % line

            spect = AudioDataset.extract_features(
                audio_path, sample_rate, truncate, window_size,
                window_stride, window, normalize_audio
            )

            yield {side: spect, side + '_path': line, 'indices': i}


class AudioSeqField(Field):
    def __init__(self, preprocessing=None, postprocessing=None,
                 include_lengths=False, batch_first=False, pad_index=0,
                 is_target=False):
        super(AudioSeqField, self).__init__(
            sequential=True, use_vocab=False, init_token=None,
            eos_token=None, fix_length=False, dtype=torch.float,
            preprocessing=preprocessing, postprocessing=postprocessing,
            lower=False, tokenize=None, include_lengths=include_lengths,
            batch_first=batch_first, pad_token=pad_index, unk_token=None,
            pad_first=False, truncate_first=False, stop_words=None,
            is_target=is_target
        )

    def pad(self, minibatch):
        assert not self.pad_first and not self.truncate_first \
               and not self.fix_length and self.sequential
        minibatch = list(minibatch)
        lengths = [x.size(1) for x in minibatch]
        max_len = max(lengths)
        nfft = minibatch[0].size(0)
        sounds = torch.full((len(minibatch), 1, nfft, max_len), self.pad_token)
        for i, (spect, len_) in enumerate(zip(minibatch, lengths)):
            sounds[i, :, :, 0:len_] = spect
        if self.include_lengths:
            return (sounds, lengths)
        return sounds

    def numericalize(self, arr, device=None):
        assert self.use_vocab is False
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=torch.int, device=device)

        if self.postprocessing is not None:
            arr = self.postprocessing(arr, None)

        if self.sequential and not self.batch_first:
            arr.permute(3, 0, 1, 2)
        if self.sequential:
            arr = arr.contiguous()

        if self.include_lengths:
            return arr, lengths
        return arr


def audio_fields(base_name, **kwargs):
    audio = AudioSeqField(pad_index=0, batch_first=True, include_lengths=True)

    return [(base_name, audio)]
