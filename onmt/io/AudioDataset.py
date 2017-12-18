# -*- coding: utf-8 -*-

import os
import codecs
import torch

from onmt.io.IO import ONMTDatasetBase, _make_example, \
                    _join_dicts, _peek, _construct_example_fromlist


class AudioDataset(ONMTDatasetBase):
    """ Dataset for data_type=='audio' """

    def sort_key(self, ex):
        "Sort using the size of the audio corpus."
        return -ex.src.size(1)

    def _process_corpus(self, fields, src_path, src_dir, tgt_path,
                        tgt_seq_length=0, tgt_seq_length_trunc=0,
                        sample_rate=0, window_size=0,
                        window_stride=0, window=None, normalize_audio=True,
                        use_filter_pred=True):
        """
        Build Example objects, Field objects, and filter_pred function
        from audio corpus.

        Args:
            fields: a dictionary of Field objects. Keys are like 'src',
                    'tgt', 'src_map', and 'alignment'.
            src_path: location of a src file containing audio paths.
            src_dir: location of source audio file.
            tgt_path: location of target-side data or None.
            tgt_seq_length: maximum target sequence length.
            tgt_seq_length_trunc: truncated target sequence length.
            sample_rate: sample rate.
            window_size: window size for spectrogram in seconds.
            window_stride: window stride for spectrogram in seconds.
            window: indow type for spectrogram generation.
            normalize_audio: subtract spectrogram by mean and divide
                             by std or not.
            use_filter_pred: use a custom filter predicate to filter
                             examples?

        Returns:
            constructed tuple of Examples objects, Field objects, filter_pred.
        """
        assert (src_dir is not None) and os.path.exists(src_dir),\
            "src_dir must be a valid directory if data_type is audio"

        self.data_type = 'audio'

        global torchaudio, librosa, np
        import torchaudio
        import librosa
        import numpy as np

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.normalize_audio = normalize_audio

        # Process the source audio corpus into examples, and process
        # the target text corpus into examples, if tgt_path is not None.
        src_examples = _read_audio_file(src_path, src_dir, "src",
                                        sample_rate, window_size,
                                        window_stride, window,
                                        normalize_audio)
        self.n_src_feats = 0

        tgt_examples, self.n_tgt_feats = \
            _make_example(tgt_path, tgt_seq_length_trunc, "tgt")

        if tgt_examples is not None:
            examples = (_join_dicts(src, tgt)
                        for src, tgt in zip(src_examples, tgt_examples))
        else:
            examples = src_examples

        # Peek at the first to see which fields are used.
        ex, examples = _peek(examples)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples)
        out_examples = (_construct_example_fromlist(ex_values, out_fields)
                        for ex_values in example_values)

        def filter_pred(example):
            if tgt_examples is not None:
                return 0 < len(example.tgt) <= tgt_seq_length
            else:
                return True

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        return out_examples, out_fields, filter_pred


def _read_audio_file(path, src_dir, side, sample_rate, window_size,
                     window_stride, window, normalize_audio, truncate=None):
    """
    Args:
        path: location of a src file containing audio paths.
        src_dir: location of source audio files.
        side: 'src' or 'tgt'.
        sample_rate: sample_rate.
        window_size: window size for spectrogram in seconds.
        window_stride: window stride for spectrogram in seconds.
        window: window type for spectrogram generation.
        normalize_audio: subtract spectrogram by mean and divide by std or not
        truncate: maximum audio length (0 or None for unlimited).

    Yields:
        a dictionary containing audio data for each line.
    """
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
            D = librosa.stft(sound, n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, window=window)
            spect, _ = librosa.magphase(D)
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
