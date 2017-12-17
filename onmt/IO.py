# -*- coding: utf-8 -*-

import os
import codecs
from collections import Counter, defaultdict
from itertools import chain, count

import torch
import torchtext.data
import torchtext.vocab

PAD_WORD = '<blank>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def __getstate__(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def __setstate__(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = __getstate__
torchtext.vocab.Vocab.__setstate__ = __setstate__


def load_fields(vocab, data_type="text"):
    vocab = dict(vocab)
    n_src_features = len(collect_features(vocab, 'src'))
    n_tgt_features = len(collect_features(vocab, 'tgt'))
    fields = get_fields(data_type, n_src_features, n_tgt_features)
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
    return fields


def collect_features(fields, side="src"):
    assert side in ["src", "tgt"]
    feats = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feats.append(key)
    return feats


def extract_features(tokens):
    """
    Args:
        tokens: A list of tokens, where each token consists of a word,
            optionally followed by u"￨"-delimited features.
    Returns:
        A sequence of words, a sequence of features, and num of features.
    """
    if not tokens:
        return [], [], -1
    split_tokens = [token.split(u"￨") for token in tokens]
    split_tokens = [token for token in split_tokens if token[0]]
    token_size = len(split_tokens[0])
    assert all(len(token) == token_size for token in split_tokens), \
        "all words must have the same number of features"
    words_and_features = list(zip(*split_tokens))
    words = words_and_features[0]
    features = words_and_features[1:]
    return words, features, token_size - 1


def merge_vocabs(vocabs, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return torchtext.vocab.Vocab(merged,
                                 specials=[PAD_WORD, BOS_WORD, EOS_WORD],
                                 max_size=vocab_size)


def make_features(batch, side, data_type='text'):
    """
    Args:
        batch (Variable): a batch of source or target data.
        side (str): for source or for target.
        data_type (str): type of the source input. Options are [text|img].
    Returns:
        A sequence of src/tgt tensors with optional feature tensors
        of size (len x batch).
    """
    assert side in ['src', 'tgt']
    if isinstance(batch.__dict__[side], tuple):
        data = batch.__dict__[side][0]
    else:
        data = batch.__dict__[side]

    feat_start = side + "_feat_"
    keys = sorted([k for k in batch.__dict__ if feat_start in k])
    features = [batch.__dict__[k] for k in keys]
    levels = [data] + features

    if data_type == 'text':
        return torch.cat([level.unsqueeze(2) for level in levels], 2)
    else:
        return levels[0]


def save_vocab(fields):
    vocab = []
    for k, f in fields.items():
        if 'vocab' in f.__dict__:
            f.vocab.stoi = dict(f.vocab.stoi)
            vocab.append((k, f.vocab))
    return vocab


def collect_feature_dicts(fields, side):
    assert side in ['src', 'tgt']
    feature_dicts = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feature_dicts.append(fields[key].vocab)
    return feature_dicts


def get_fields(data_type, n_src_features, n_tgt_features):
    """
    Args:
        data_type: type of the source input. Options are [text|img|audio].
        n_src_features: the number of source features to create Field for.
        n_tgt_features: the number of target features to create Field for.
    Returns:
        A dictionary whose keys are strings and whose values are the
        corresponding Field objects.
    """
    fields = {}
    if data_type == 'text':
        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)

    elif data_type == 'img':
        def make_img(data, _):
            c = data[0].size(0)
            h = max([t.size(1) for t in data])
            w = max([t.size(2) for t in data])
            imgs = torch.zeros(len(data), c, h, w)
            for i, img in enumerate(data):
                imgs[i, :, 0:img.size(1), 0:img.size(2)] = img
            return imgs

        fields["src"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=make_img, sequential=False)

    elif data_type == 'audio':
        def make_audio(data, _):
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

    def make_src(data, _):
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

    def make_tgt(data, _):
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


def build_dataset(fields, data_type, src_path, tgt_path, src_dir=None,
                  src_seq_length=0, tgt_seq_length=0,
                  src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                  dynamic_dict=True, sample_rate=0,
                  window_size=0, window_stride=0, window=None,
                  normalize_audio=True, use_filter_pred=True):

    if data_type == 'text':
        dataset = TextDataset(fields, src_path, tgt_path,
                              src_seq_length=src_seq_length,
                              tgt_seq_length=tgt_seq_length,
                              src_seq_length_trunc=src_seq_length_trunc,
                              tgt_seq_length_trunc=tgt_seq_length_trunc,
                              dynamic_dict=dynamic_dict,
                              use_filter_pred=use_filter_pred)
    elif data_type == 'img':
        dataset = ImageDataset(fields, src_path, src_dir, tgt_path,
                               tgt_seq_length=tgt_seq_length,
                               tgt_seq_length_trunc=tgt_seq_length_trunc,
                               use_filter_pred=use_filter_pred)

    elif data_type == 'audio':
        dataset = AudioDataset(fields, src_path, src_dir, tgt_path,
                               tgt_seq_length=tgt_seq_length,
                               tgt_seq_length_trunc=tgt_seq_length_trunc,
                               sample_rate=sample_rate,
                               window_size=window_size,
                               window_stride=window_stride,
                               window=window,
                               normalize_audio=normalize_audio,
                               use_filter_pred=use_filter_pred)

    return dataset


def build_vocab(train, data_type, share_vocab,
                src_vocab_size, src_words_min_frequency,
                tgt_vocab_size, tgt_words_min_frequency):
    """
    Args:
        train: a train dataset.
        data_type: "text", "img" or "audio"?
        share_vocab(bool): share source and target vocabulary?
        src_vocab_size(int): size of the source vocabulary.
        src_words_min_frequency(int): the minimum frequency needed to
                include a source word in the vocabulary.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.
    """
    fields = train.fields

    fields["tgt"].build_vocab(train, max_size=tgt_vocab_size,
                              min_freq=tgt_words_min_frequency)
    for j in range(train.n_tgt_feats):
        fields["tgt_feat_" + str(j)].build_vocab(train)

    if data_type == 'text':
        fields["src"].build_vocab(train, max_size=src_vocab_size,
                                  min_freq=src_words_min_frequency)
        for j in range(train.n_src_feats):
            fields["src_feat_" + str(j)].build_vocab(train)

        # Merge the input and output vocabularies.
        if share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            merged_vocab = merge_vocabs(
                [fields["src"].vocab, fields["tgt"].vocab],
                vocab_size=src_vocab_size)
            fields["src"].vocab = merged_vocab
            fields["tgt"].vocab = merged_vocab


def _join_dicts(*args):
    """
    Args:
        dictionaries with disjoint keys.
    Returns:
        a single dictionary that has the union of these keys.
    """
    return dict(chain(*[d.items() for d in args]))


def _peek(seq):
    """
    Args:
        seq: an iterator.

    Returns:
        the first thing returned by calling next() on the iterator
        and an iterator created by re-chaining that value to the beginning
        of the iterator.
    """
    first = next(seq)
    return first, chain([first], seq)


def _construct_example_fromlist(data, fields):
    ex = torchtext.data.Example()
    for (name, field), val in zip(fields, data):
        if field is not None:
            setattr(ex, name, field.preprocess(val))
        else:
            setattr(ex, name, val)
    return ex


def _read_text_file(path, truncate, side):
    """
    Args:
        path: location of a src or tgt file.
        truncate: maximum sequence length (0 for unlimited).

    Yields:
        (word, features, nfeat) triples for each line.
    """
    with codecs.open(path, "r", "utf-8") as corpus_file:
        for i, line in enumerate(corpus_file):
            line = line.strip().split()
            if truncate:
                line = line[:truncate]
            words, feats, n_feats = extract_features(line)
            example_dict = {side: words, "indices": i}
            if feats:
                prefix = side + "_feat_"
                example_dict.update((prefix + str(j), f)
                                    for j, f in enumerate(feats))
            yield example_dict, n_feats


def _make_example(path, truncate, side):
    """
    Process the text corpus into (examples, num_feats) tuple.
    """
    assert side in ['src', 'tgt']

    if path is None:
        return (None, 0)

    examples = _read_text_file(path, truncate, side)
    (_, num_feats), examples = _peek(examples)

    out_examples = (ex for ex, nfeats in examples)

    return (out_examples, num_feats)


def _read_img_file(path, src_dir, side, truncate=None):
    """
    Args:
        path: location of a src file containing image paths
        src_dir: location of source images
        side: 'src' or 'tgt'
        truncate: maximum img size ((0,0) or None for unlimited)

    Yields:
        a dictionary containing image data, path and index for each line.
    """
    with codecs.open(path, "r", "utf-8") as corpus_file:
        index = 0
        for line in corpus_file:
            img_path = os.path.join(src_dir, line.strip())
            if not os.path.exists(img_path):
                img_path = line
            assert os.path.exists(img_path), \
                'img path %s not found' % (line.strip())
            img = transforms.ToTensor()(Image.open(img_path))
            if truncate and truncate != (0, 0):
                if not (img.size(1) <= truncate[0]
                        and img.size(2) <= truncate[1]):
                    continue
            example_dict = {side: img,
                            side+'_path': line.strip(),
                            'indices': index}
            index += 1
            yield example_dict


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
        image for each line.
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


class OrderedIterator(torchtext.data.Iterator):
    def create_batches(self):
        if self.train:
            self.batches = torchtext.data.pool(
                self.data(), self.batch_size,
                self.sort_key, self.batch_size_fn,
                random_shuffler=self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class ONMTDatasetBase(torchtext.data.Dataset):
    """
    A dataset basically supports iteration over all the examples
    it contains. We currently have 3 datasets inheriting this base
    for 3 types of corpus respectively: "text", "img", "audio".

    Internally it initializes an `torchtext.data.Dataset` object with
    the following attributes:

     `examples`: a sequence of `torchtext.data.Example` objects.
     `fields`: a dictionary associating str keys with Field objects. Does not
            necessarily have the same keys as the input fields.
    """
    def __init__(self, *args, **kwargs):
        examples, fields, filter_pred = self._process_corpus(*args, **kwargs)
        super(ONMTDatasetBase, self).__init__(
            examples, fields, filter_pred
        )

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(ONMTDatasetBase, self).__reduce_ex__()

    def collapse_copy_scores(self, scores, batch, tgt_vocab):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            index = batch.indices.data[b]
            src_vocab = self.src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    scores[:, b, ti] += scores[:, b, offset + i]
                    scores[:, b, offset + i].fill_(1e-20)
        return scores


class TextDataset(ONMTDatasetBase):
    """ Dataset for data_type=='text' """

    def sort_key(self, ex):
        "Sort using the size of source example."
        return -len(ex.src)

    def _process_corpus(self, fields, src_path, tgt_path,
                        src_seq_length=0, tgt_seq_length=0,
                        src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                        dynamic_dict=True, use_filter_pred=True):
        """
        Build Example objects, Field objects, and filter_pred function
        from text corpus.

        Args:
            fields: a dictionary of Field objects. Keys are like 'src',
                    'tgt', 'src_map', and 'alignment'.
            src_path: location of source-side data.
            tgt_path: location of target-side data or None. If should be the
                      same length as the source-side data if it exists.
            src_seq_length: maximum source sequence length.
            tgt_seq_length: maximum target sequence length.
            src_seq_length_trunc: truncated source sequence length.
            tgt_seq_length_trunc: truncated target sequence length.
            dynamic_dict: create dynamic dictionaries?
            use_filter_pred: use a custom filter predicate to filter examples?

        Returns:
            constructed tuple of Examples objects, Field objects, filter_pred.
        """
        self.data_type = 'text'

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []

        # Process the corpus into examples, and extract number of features,
        # if any. Note tgt_path might be None.
        src_examples, self.n_src_feats = \
            _make_example(src_path, src_seq_length_trunc, "src")

        tgt_examples, self.n_tgt_feats = \
            _make_example(tgt_path, tgt_seq_length_trunc, "tgt")

        # examples: one for each src line or (src, tgt) line pair.
        # Each element is a dictionary whose keys represent at minimum
        # the src tokens and their indices and potentially also the
        # src and tgt features and alignment information.
        if tgt_examples is not None:
            examples = (_join_dicts(src, tgt)
                        for src, tgt in zip(src_examples, tgt_examples))
        else:
            examples = src_examples

        if dynamic_dict:
            examples = self._dynamic_dict(examples)

        # Peek at the first to see which fields are used.
        ex, examples = _peek(examples)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples)
        out_examples = (_construct_example_fromlist(ex_values, out_fields)
                        for ex_values in example_values)

        def filter_pred(example):
            return 0 < len(example.src) <= src_seq_length \
               and 0 < len(example.tgt) <= tgt_seq_length

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        return out_examples, out_fields, filter_pred

    def _dynamic_dict(self, examples):
        for example in examples:
            src = example["src"]
            src_vocab = torchtext.vocab.Vocab(Counter(src))
            self.src_vocabs.append(src_vocab)
            # Mapping source tokens to indices in the dynamic dict.
            src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
            example["src_map"] = src_map

            if "tgt" in example:
                tgt = example["tgt"]
                mask = torch.LongTensor(
                        [0] + [src_vocab.stoi[w] for w in tgt] + [0])
                example["alignment"] = mask
            yield example


class ImageDataset(ONMTDatasetBase):
    """ Dataset for data_type=='img' """

    def sort_key(self, ex):
        "Sort using the size of the image."
        return (-ex.src.size(2), -ex.src.size(1))

    def _process_corpus(self, fields, src_path, src_dir, tgt_path,
                        tgt_seq_length=0, tgt_seq_length_trunc=0,
                        use_filter_pred=True):
        """
        Build Example objects, Field objects, and filter_pred function
        from image corpus.

        Args:
            fields: a dictionary of Field objects. Keys are like 'src',
                    'tgt', 'src_map', and 'alignment'.
            src_path: location of a src file containing image paths
            src_dir: location of source images
            tgt_path: location of target-side data or None.
            tgt_seq_length: maximum target sequence length.
            tgt_seq_length_trunc: truncated target sequence length.
            use_filter_pred: use a custom filter predicate to filter examples?

        Returns:
            constructed tuple of Examples objects, Field objects, filter_pred.
        """
        assert (src_dir is not None) and os.path.exists(src_dir),\
            'src_dir must be a valid directory if data_type is img'

        self.data_type = 'img'

        global Image, transforms
        from PIL import Image
        from torchvision import transforms

        # Process the source image corpus into examples, and process
        # the target text corpus into examples, if tgt_path is not None.
        src_examples = _read_img_file(src_path, src_dir, "src")
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
