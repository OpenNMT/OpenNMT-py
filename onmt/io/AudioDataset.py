# -*- coding: utf-8 -*-

import codecs
import os

import torch
import torchtext

from onmt.io.DatasetBase import ONMTDatasetBase, PAD_WORD, BOS_WORD, EOS_WORD


class AudioDataset(ONMTDatasetBase):
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
    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 num_src_feats=0, num_tgt_feats=0,
                 tgt_seq_length=0, sample_rate=0,
                 window_size=0.0, window_stride=0.0, window=None,
                 normalize_audio=True, use_filter_pred=True):
        self.data_type = 'audio'

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.normalize_audio = normalize_audio

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats

        if tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)
        out_examples = (self._construct_example_fromlist(
                            ex_values, out_fields)
                        for ex_values in example_values)
        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        out_examples = list(out_examples)

        def filter_pred(example):
            if tgt_examples_iter is not None:
                return 0 < len(example.tgt) <= tgt_seq_length
            else:
                return True

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(AudioDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    def sort_key(self, ex):
        """ Sort using duration time of the sound spectrogram. """
        return ex.src.size(1)

    @staticmethod
    def make_audio_examples_nfeats_tpl(path, audio_dir,
                                       sample_rate, window_size,
                                       window_stride, window,
                                       normalize_audio, truncate=None):
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
        examples_iter = AudioDataset.read_audio_file(
                path, audio_dir, "src", sample_rate,
                window_size, window_stride, window,
                normalize_audio, truncate)
        num_feats = 0  # Source side(audio) has no features.

        return (examples_iter, num_feats)

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

        global torchaudio, librosa, np
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

    @staticmethod
    def get_fields(n_src_features, n_tgt_features):
        """
        Args:
            n_src_features: the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features: the number of target features to
                create `torchtext.data.Field` for.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        def make_audio(data, vocab, is_train):
            nfft = data[0].size(0)
            t = max([t.size(1) for t in data])
            sounds = torch.zeros(len(data), 1, nfft, t)
            for i, spect in enumerate(data):
                sounds[i, :, :, 0:spect.size(1)] = spect
            return sounds

        fields["src"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=make_audio, sequential=False)

        for j in range(n_src_features):
            fields["src_feat_"+str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        for j in range(n_tgt_features):
            fields["tgt_feat_"+str(j)] = \
                torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD,
                                     pad_token=PAD_WORD)

        def make_src(data, vocab, is_train):
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(src_size, len(data), src_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=make_src, sequential=False)

        def make_tgt(data, vocab, is_train):
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[:sent.size(0), i] = sent
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            postprocessing=make_tgt, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            sequential=False)

        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        For audio corpus, source side is in form of audio, thus
        no feature; while target side is in form of text, thus
        we can extract its text features.

        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.

        Returns:
            number of features on `side`.
        """
        if side == 'src':
            num_feats = 0
        else:
            with codecs.open(corpus_file, "r", "utf-8") as cf:
                f_line = cf.readline().strip().split()
                _, _, num_feats = AudioDataset.extract_text_features(f_line)

        return num_feats
