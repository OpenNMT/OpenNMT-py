# -*- coding: utf-8 -*-

import os

from onmt.io import ONMTDatasetBase, _make_example, \
                    _join_dicts, _peek, _construct_example_fromlist, \
                    _read_audio_file


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
