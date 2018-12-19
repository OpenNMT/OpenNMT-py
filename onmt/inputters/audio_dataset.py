# -*- coding: utf-8 -*-
import codecs
import os
from tqdm import tqdm

import torch

from onmt.inputters.dataset_base import DatasetBase


class AudioDataset(DatasetBase):
    data_type = 'audio'  # get rid of this class attribute asap

    @staticmethod
    def sort_key(ex):
        """ Sort using duration time of the sound spectrogram. """
        return ex.src.size(1)

    @staticmethod
    def extract_features(audio_path, sample_rate, truncate, window_size,
                         window_stride, window, normalize_audio):
        global torchaudio, librosa, np
        import torchaudio
        import librosa
        import numpy as np

        sound, sample_rate_ = torchaudio.load(audio_path)
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
        path,
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
            path (str): location of a src file containing audio paths.
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
        assert isinstance(path, str), "Iterators not supported for audio"
        assert src_dir is not None and os.path.exists(src_dir),\
            "src_dir must be a valid directory if data_type is audio"

        with codecs.open(path, "r", "utf-8") as corpus_file:
            for i, line in enumerate(tqdm(corpus_file)):
                audio_path = os.path.join(src_dir, line.strip())
                if not os.path.exists(audio_path):
                    audio_path = line.strip()

                assert os.path.exists(audio_path), \
                    'audio path %s not found' % (line.strip())

                spect = AudioDataset.extract_features(
                    audio_path, sample_rate, truncate, window_size,
                    window_stride, window, normalize_audio
                )

                yield {side: spect, side + '_path': line.strip(),
                       side + '_lengths': spect.size(1), 'indices': i}
