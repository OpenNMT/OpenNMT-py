# -*- coding: utf-8 -*-
"""
    AudioDataset
"""
import codecs
import os

import torch

from onmt.inputters.dataset_base import DatasetBase


class AudioDataset(DatasetBase):
    """ Dataset for data_type=='audio'

        Build `Example` objects, `Field` objects, and filter_pred function
        from audio corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            tgt_seq_length (int): maximum target sequence length.
            sample_rate (int): sample rate.
            window_size (float): window size for spectrogram in seconds.
            window_stride (float): window stride for spectrogram in seconds.
            window (str): window type for spectrogram generation.
            normalize_audio (bool): subtract spectrogram by mean and divide
                by std or not.
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """
    data_type = 'audio'

    @staticmethod
    def sort_key(ex):
        """ Sort using duration time of the sound spectrogram. """
        return ex.src.size(1)

    @classmethod
    def make_examples_nfeats_tpl(cls, path, directory, sample_rate,
                                 window_size, window_stride, window,
                                 normalize_audio, truncate=None, **kwargs):
        """
        Args:
            path (str): location of a src file containing audio paths.
            audio_dir (str): location of source audio files.
            sample_rate (int): sample_rate.
            window_size (float) : window size for spectrogram in seconds.
            window_stride (float): window stride for spectrogram in seconds.
            window (str): window type for spectrogram generation.
            normalize_audio (bool): subtract spectrogram by mean and divide
                by std or not.
            truncate (int): maximum audio length (0 or None for unlimited).

        Returns:
            (example_dict iterator, num_feats) tuple
        """
        if path is None:
            raise ValueError("AudioDataset requires a non None path")
        examples_iter = cls.read_audio_file(
            path, directory, "src", sample_rate,
            window_size, window_stride, window,
            normalize_audio, truncate)

        return examples_iter, 0

    @classmethod
    def read_audio_file(cls, path, src_dir, side, sample_rate, window_size,
                        window_stride, window, normalize_audio,
                        truncate=None):
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
        # might be the case that this needs to be an instance method
        assert src_dir is not None and os.path.exists(src_dir), \
            "src_dir must be a valid directory if data_type is audio"

        import torchaudio
        import librosa
        import numpy as np

        with codecs.open(path, "r", "utf-8") as corpus_file:
            index = 0
            for line in corpus_file:
                audio_path = os.path.join(src_dir, line.strip())
                if not os.path.exists(audio_path):
                    audio_path = line

                assert os.path.exists(audio_path), \
                    'audio path %s not found' % (line.strip())

                sound, sample_rate = torchaudio.load(audio_path)
                if truncate and truncate > 0:
                    if sound.size(0) > truncate:
                        continue

                # TODO: find out what this was supposed to be
                assert sample_rate == sample_rate, \
                    'Sample rate of %s != -sample_rate (%d vs %d)' \
                    % (audio_path, sample_rate, sample_rate)

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

                example_dict = {side: spect,
                                side + '_path': line.strip(),
                                'indices': index}
                index += 1

                yield example_dict
