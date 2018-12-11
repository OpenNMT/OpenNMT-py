# -*- coding: utf-8 -*-
import codecs
import os
import sys
import io
from tqdm import tqdm

import torch

from onmt.inputters.dataset_base import NonTextDatasetBase


class AudioDataset(NonTextDatasetBase):
    data_type = 'audio'  # get rid of this class attribute asap

    @staticmethod
    def sort_key(ex):
        """ Sort using duration time of the sound spectrogram. """
        return ex.src.size(1)

    @staticmethod
    def make_audio_examples(path, audio_dir, sample_rate, window_size,
                            window_stride, window, normalize_audio,
                            truncate=None):
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
            example_dict iterator
        """
        examples_iter = AudioDataset.read_audio_file(
            path, audio_dir, "src", sample_rate,
            window_size, window_stride, window,
            normalize_audio, truncate)

        return examples_iter

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

    @staticmethod
    def read_audio_file(path, src_dir, side, sample_rate, window_size,
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
        assert (src_dir is not None) and os.path.exists(src_dir),\
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


class ShardedAudioCorpusIterator(object):
    """
    This is the iterator for audio corpus, used for sharding large audio
    corpus into small shards, to avoid hogging memory.

    Inside this iterator, it automatically divides the audio files into
    shards of size `shard_size`. Then, for each shard, it processes
    into (example_dict, n_features) tuples when iterates.
    """
    def __init__(self, src_dir, corpus_path, truncate, side, shard_size,
                 sample_rate, window_size, window_stride,
                 window, normalize_audio=True, assoc_iter=None):
        """
        Args:
            src_dir: the directory containing audio files
            corpus_path: the path containing audio file names
            truncate: maximum audio length (0 or None for unlimited).
            side: "src" or "tgt".
            shard_size: the shard size, 0 means not sharding the file.
            sample_rate (int): sample_rate.
            window_size (float) : window size for spectrogram in seconds.
            window_stride (float): window stride for spectrogram in seconds.
            window (str): window type for spectrogram generation.
            normalize_audio (bool): subtract spectrogram by mean and divide
                by std or not.
            assoc_iter: if not None, it is the associate iterator that
                        this iterator should align its step with.
        """
        try:
            # The codecs module seems to have bugs with seek()/tell(),
            # so we use io.open().
            self.corpus = io.open(corpus_path, "r", encoding="utf-8")
        except IOError:
            sys.stderr.write("Failed to open corpus file: %s" % corpus_path)
            sys.exit(1)

        self.side = side
        self.src_dir = src_dir
        self.shard_size = shard_size
        self.sample_rate = sample_rate
        self.truncate = truncate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.normalize_audio = normalize_audio
        self.assoc_iter = assoc_iter
        self.last_pos = 0
        self.last_line_index = -1
        self.line_index = -1
        self.eof = False

    def __iter__(self):
        """
        Iterator of (example_dict, nfeats).
        On each call, it iterates over as many (example_dict, nfeats) tuples
        until this shard's size equals to or approximates `self.shard_size`.
        """
        iteration_index = -1
        if self.assoc_iter is not None:
            # We have associate iterator, just yields tuples
            # util we run parallel with it.
            while self.line_index < self.assoc_iter.line_index:
                line = self.corpus.readline()
                assert line != '', "The corpora must have same number of lines"

                self.line_index += 1
                iteration_index += 1
                yield self._example_dict_iter(line, iteration_index)

            if self.assoc_iter.eof:
                self.eof = True
                self.corpus.close()
        else:
            # Yield tuples until this shard's size reaches the threshold.
            self.corpus.seek(self.last_pos)
            while True:
                if self.shard_size != 0 and self.line_index % 64 == 0:
                    cur_pos = self.corpus.tell()
                    if self.line_index \
                            >= self.last_line_index + self.shard_size:
                        self.last_pos = cur_pos
                        self.last_line_index = self.line_index
                        raise StopIteration

                line = self.corpus.readline()
                if line == '':
                    self.eof = True
                    self.corpus.close()
                    raise StopIteration

                self.line_index += 1
                iteration_index += 1
                yield self._example_dict_iter(line, iteration_index)

    def hit_end(self):
        return self.eof

    def _example_dict_iter(self, line, index):
        line = line.strip()
        audio_path = os.path.join(self.src_dir, line)
        if not os.path.exists(audio_path):
            audio_path = line

        assert os.path.exists(audio_path), 'audio path %s not found' % line

        spect = AudioDataset.extract_features(
            audio_path, self.sample_rate, self.truncate, self.window_size,
            self.window_stride, self.window, self.normalize_audio
        )
        return {self.side: spect, self.side + '_path': line,
                self.side + '_lengths': spect.size(1), 'indices': index}
