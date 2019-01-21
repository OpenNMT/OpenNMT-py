# -*- coding: utf-8 -*-
import os
from tqdm import tqdm

import torch

from onmt.datatypes.datareader_base import DataReaderBase

# imports of datatype-specific dependencies
try:
    import torchaudio
    import librosa
    import numpy as np
except ImportError:
    torchaudio, librosa, np = None, None, None


class AudioDataReader(DataReaderBase):
    """Create a dataset reader.

    Args:
        sample_rate (int): sample_rate.
        window_size (float) : window size for spectrogram in seconds.
        window_stride (float): window stride for spectrogram in seconds.
        window (str): window type for spectrogram generation.
        normalize_audio (bool): subtract spectrogram by mean and divide
            by std or not.
        truncate (int or NoneType): maximum audio length
            (0 or None for unlimited).
    """

    def __init__(self, sample_rate, window_size,
                 window_stride, window, normalize_audio, truncate=None):
        self._check_deps()
        super(AudioDataReader, self).__init__()
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.normalize_audio = normalize_audio
        self.truncate = truncate

    @classmethod
    def from_opt(cls, opt):
        """Alternative constructor."""
        return cls(opt.sample_rate, opt.window_size,
                   opt.window_stride, opt.window, True, None)

    @staticmethod
    def _check_deps():
        if any([torchaudio is None, librosa is None, np is None]):
            AudioDataReader._raise_missing_dep(
                "torchaudio", "librosa", "numpy")

    @staticmethod
    def sort_key(ex):
        """ Sort using duration time of the sound spectrogram. """
        return ex.src.size(1)

    def extract_features(self, audio_path):
        # torchaudio loading options recently changed. It's probably
        # straightforward to rewrite the audio handling to make use of
        # up-to-date torchaudio, but in the meantime there is a legacy
        # method which uses the old defaults
        sound, sample_rate_ = torchaudio.legacy.load(audio_path)
        if self.truncate and self.truncate > 0:
            if sound.size(0) > self.truncate:
                sound = sound[:self.truncate]

        assert sample_rate_ == self.sample_rate, \
            'Sample rate of %s != -sample_rate (%d vs %d)' \
            % (audio_path, sample_rate_, self.sample_rate)

        sound = sound.numpy()
        if len(sound.shape) > 1:
            if sound.shape[1] == 1:
                sound = sound.squeeze()
            else:
                sound = sound.mean(axis=1)  # average multiple channels

        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        d = librosa.stft(sound, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, _ = librosa.magphase(d)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize_audio:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
        return spect

    def read(self, audio_files, side, src_dir=None):
        """Read sound files.

        Args:
            audio_files (str, List[str]): Either a file of one file path per
                line (either existing path or path relative to `src_dir`)
                or a list thereof.
            side (str): 'src' or 'tgt'.
            src_dir (str): location of source audio files.

        Yields:
            a dictionary containing audio data for each line.
        """

        assert src_dir is not None and os.path.exists(src_dir),\
            "src_dir must be a valid directory if data_type is audio"

        if isinstance(audio_files, str):
            audio_files = self._read_file(audio_files)

        for i, line in enumerate(tqdm(audio_files)):
            filename = line.strip()
            audio_path = os.path.join(src_dir, filename)
            if not os.path.exists(audio_path):
                audio_path = filename

            assert os.path.exists(audio_path), \
                'audio path %s not found' % (line.strip())

            spect = self.extract_features(audio_path)

            yield {side: spect, side + '_path': line.strip(),
                   side + '_lengths': spect.size(1), 'indices': i}
