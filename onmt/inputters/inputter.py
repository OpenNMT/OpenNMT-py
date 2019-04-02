# -*- coding: utf-8 -*-
"""
    Defining general functions for inputters
"""
import glob
import os

from collections import Counter, defaultdict, OrderedDict
from itertools import count

import torch
import torchtext.data
import torchtext.vocab
from torch.autograd import Variable

from onmt.inputters.dataset_base import UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD
from onmt.inputters.text_dataset import TextDataset
from onmt.inputters.amr_dataset import AMRDataset
from onmt.inputters.image_dataset import ImageDataset
from onmt.inputters.audio_dataset import AudioDataset
from onmt.utils.logging import logger

import gc
import numpy as np
from tree import read_tree

def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = _getstate
torchtext.vocab.Vocab.__setstate__ = _setstate


def get_fields(data_type, n_src_features, n_tgt_features):
    """
    Args:
        data_type: type of the source input. Options are [text|img|audio].
        n_src_features: the number of source features to
            create `torchtext.data.Field` for.
        n_tgt_features: the number of target features to
            create `torchtext.data.Field` for.

    Returns:
        A dictionary whose keys are strings and whose values are the
        corresponding Field objects.
    """
    if data_type == 'text':
        return TextDataset.get_fields(n_src_features, n_tgt_features)
    elif data_type == 'img':
        return ImageDataset.get_fields(n_src_features, n_tgt_features)
    elif data_type == 'audio':
        return AudioDataset.get_fields(n_src_features, n_tgt_features)
    elif data_type == 'amr':
        return AMRDataset.get_fields(n_src_features, n_tgt_features)
    else:
        raise ValueError("Data type not implemented")


def load_fields_from_vocab(vocab, data_type):
    """
    Load Field objects from `vocab.pt` file.
    """
    vocab = dict(vocab)
    n_src_features = len(collect_features(vocab, 'src'))
    n_tgt_features = len(collect_features(vocab, 'tgt'))
    fields = get_fields(data_type, n_src_features, n_tgt_features)
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
    return fields


def save_fields_to_vocab(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    """
    vocab = []
    for k, f in fields.items():
        if f is not None and 'vocab' in f.__dict__:
            f.vocab.stoi = f.vocab.stoi
            vocab.append((k, f.vocab))
    return vocab


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
                                 specials=[UNK_WORD, PAD_WORD,
                                           BOS_WORD, EOS_WORD],
                                 max_size=vocab_size)


def get_num_features(data_type, corpus_file, side):
    """
    Args:
        data_type (str): type of the source input.
            Options are [text|img|audio].
        corpus_file (str): file path to get the features.
        side (str): for source or for target.

    Returns:
        number of features on `side`.
    """
    assert side in ["src", "tgt"]

    if data_type == 'text':
        return TextDataset.get_num_features(corpus_file, side)
    elif data_type == 'img':
        return ImageDataset.get_num_features(corpus_file, side)
    elif data_type == 'audio':
        return AudioDataset.get_num_features(corpus_file, side)
    elif data_type == 'amr':
        return AMRDataset.get_num_features(corpus_file, side)
    else:
        raise ValueError("Data type not implemented")


def make_features(batch, side, data_type):
    """
    Args:
        batch (Tensor): a batch of source or target data.
        side (str): for source or for target.
        data_type (str): type of the source input.
            Options are [text|img|audio].
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
    
    if side == 'src' and 'src_adj' in batch.__dict__.keys():
        adj = batch.__dict__['src_adj']
    if side == 'src' and 'src_parents' in batch.__dict__.keys():
        parents = batch.__dict__['src_parents']
        
    if data_type == 'amr':
        if side == 'tgt':
            return torch.cat([level.unsqueeze(2) for level in levels], 2)
        return (torch.cat([level.unsqueeze(2) for level in levels], 2),
                adj,
                parents)
    elif data_type == 'text':
        return torch.cat([level.unsqueeze(2) for level in levels], 2)
    else:
        return levels[0]


def collect_features(fields, side="src"):
    """
    Collect features from Field object.
    """
    assert side in ["src", "tgt"]
    feats = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feats.append(key)
    return feats


def collect_feature_vocabs(fields, side):
    """
    Collect feature Vocab objects from Field object.
    """
    assert side in ['src', 'tgt']
    feature_vocabs = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feature_vocabs.append(fields[key].vocab)
    return feature_vocabs


def build_dataset(fields, data_type, src_data_iter=None, src_path=None,
                  src_dir=None, tgt_data_iter=None, tgt_path=None,
                  src_seq_length=0, tgt_seq_length=0,
                  src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                  dynamic_dict=True, sample_rate=0,
                  window_size=0, window_stride=0, window=None,
                  normalize_audio=True, use_filter_pred=True,
                  image_channel_size=3, reentrancies=False):
    """
    Build src/tgt examples iterator from corpus files, also extract
    number of features.
    """

    def _make_examples_nfeats_tpl(data_type, src_data_iter, src_path, src_dir,
                                  src_seq_length_trunc, sample_rate,
                                  window_size, window_stride,
                                  window, normalize_audio, reentrancies,
                                  image_channel_size=3):
        """
        Process the corpus into (example_dict iterator, num_feats) tuple
        on source side for different 'data_type'.
        """
        if data_type == 'text':
            src_examples_iter, num_src_feats = \
                TextDataset.make_text_examples_nfeats_tpl(
                    src_data_iter, src_path, src_seq_length_trunc, "src")

        elif data_type == 'img':
            src_examples_iter, num_src_feats = \
                ImageDataset.make_image_examples_nfeats_tpl(
                    src_data_iter, src_path, src_dir, image_channel_size)

        elif data_type == 'audio':
            if src_data_iter:
                raise ValueError("""Data iterator for AudioDataset isn't
                                    implemented""")

            if src_path is None:
                raise ValueError("AudioDataset requires a non None path")
            src_examples_iter, num_src_feats = \
                AudioDataset.make_audio_examples_nfeats_tpl(
                    src_path, src_dir, sample_rate,
                    window_size, window_stride, window,
                    normalize_audio)
                
        elif data_type == 'amr':
            src_examples_iter, num_src_feats = \
                AMRDataset.make_amr_examples_nfeats_tpl(
                    src_data_iter, src_path, src_seq_length_trunc, 
                    "src", reentrancies)

        return src_examples_iter, num_src_feats

    src_examples_iter, num_src_feats = \
        _make_examples_nfeats_tpl(data_type, src_data_iter, src_path, src_dir,
                                  src_seq_length_trunc, sample_rate,
                                  window_size, window_stride,
                                  window, normalize_audio, reentrancies,
                                  image_channel_size=image_channel_size)

    # For all data types, the tgt side corpus is in form of text.
    tgt_examples_iter, num_tgt_feats = \
        TextDataset.make_text_examples_nfeats_tpl(
            tgt_data_iter, tgt_path, tgt_seq_length_trunc, "tgt")

    if data_type == 'text':
        dataset = TextDataset(fields, src_examples_iter, tgt_examples_iter,
                              num_src_feats, num_tgt_feats,
                              src_seq_length=src_seq_length,
                              tgt_seq_length=tgt_seq_length,
                              dynamic_dict=dynamic_dict,
                              use_filter_pred=use_filter_pred)

    elif data_type == 'img':
        dataset = ImageDataset(fields, src_examples_iter, tgt_examples_iter,
                               num_src_feats, num_tgt_feats,
                               tgt_seq_length=tgt_seq_length,
                               use_filter_pred=use_filter_pred,
                               image_channel_size=image_channel_size)

    elif data_type == 'audio':
        dataset = AudioDataset(fields, src_examples_iter, tgt_examples_iter,
                               tgt_seq_length=tgt_seq_length,
                               use_filter_pred=use_filter_pred)
    elif data_type == 'amr':
        dataset = AMRDataset(fields, src_examples_iter, tgt_examples_iter,
                              num_src_feats, num_tgt_feats,
                              src_seq_length=src_seq_length,
                              tgt_seq_length=tgt_seq_length,
                              dynamic_dict=dynamic_dict,
                              use_filter_pred=use_filter_pred)        

    return dataset


def _build_field_vocab(field, counter, **kwargs):
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token]
        if tok is not None))
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)


def build_vocab(train_dataset_files, fields, data_type, share_vocab,
                src_vocab_path, src_vocab_size, src_words_min_frequency,
                tgt_vocab_path, tgt_vocab_size, tgt_words_min_frequency):
    """
    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict): fields to build vocab for.
        data_type: "text", "img" or "audio"?
        share_vocab(bool): share source and target vocabulary?
        src_vocab_path(string): Path to src vocabulary file.
        src_vocab_size(int): size of the source vocabulary.
        src_words_min_frequency(int): the minimum frequency needed to
                include a source word in the vocabulary.
        tgt_vocab_path(string): Path to tgt vocabulary file.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.

    Returns:
        Dict of Fields
    """
    counter = {}

    # Prop src from field to get lower memory using when training with image
    if data_type == 'img' or data_type == 'audio':
        fields.pop("src")

    for k in fields:
        counter[k] = Counter()

    # Load vocabulary
    src_vocab = load_vocabulary(src_vocab_path, tag="source")
    tgt_vocab = load_vocabulary(tgt_vocab_path, tag="target")

    for index, path in enumerate(train_dataset_files):
        dataset = torch.load(path)
        logger.info(" * reloading %s." % path)
        for ex in dataset.examples:
            for k in fields:
                val = getattr(ex, k, None)
                if not fields[k].sequential:
                    continue
                elif k == 'src' and src_vocab:
                    val = [item for item in val if item in src_vocab]
                elif k == 'tgt' and tgt_vocab:
                    val = [item for item in val if item in tgt_vocab]
                counter[k].update(val)

        # Drop the none-using from memory but keep the last
        if (index < len(train_dataset_files) - 1):
            dataset.examples = None
            gc.collect()
            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

    _build_field_vocab(fields["tgt"], counter["tgt"],
                       max_size=tgt_vocab_size,
                       min_freq=tgt_words_min_frequency)
    logger.info(" * tgt vocab size: %d." % len(fields["tgt"].vocab))

    # All datasets have same num of n_tgt_features,
    # getting the last one is OK.
    for j in range(dataset.n_tgt_feats):
        key = "tgt_feat_" + str(j)
        _build_field_vocab(fields[key], counter[key])
        logger.info(" * %s vocab size: %d." % (key,
                                               len(fields[key].vocab)))

    if data_type == 'text' or data_type == 'amr':
        _build_field_vocab(fields["src"], counter["src"],
                           max_size=src_vocab_size,
                           min_freq=src_words_min_frequency)
        logger.info(" * src vocab size: %d." % len(fields["src"].vocab))     

        # All datasets have same num of n_src_features,
        # getting the last one is OK.
        for j in range(dataset.n_src_feats):
            key = "src_feat_" + str(j)
            _build_field_vocab(fields[key], counter[key])
            logger.info(" * %s vocab size: %d." %
                        (key, len(fields[key].vocab)))

        # Merge the input and output vocabularies.
        if share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            logger.info(" * merging src and tgt vocab...")
            merged_vocab = merge_vocabs(
                [fields["src"].vocab, fields["tgt"].vocab],
                vocab_size=src_vocab_size)
            fields["src"].vocab = merged_vocab
            fields["tgt"].vocab = merged_vocab

    return fields


def load_vocabulary(vocabulary_path, tag=""):
    """
    Loads a vocabulary from the given path.
    :param vocabulary_path: path to load vocabulary from
    :param tag: tag for vocabulary (only used for logging)
    :return: vocabulary or None if path is null
    """
    vocabulary = None
    if vocabulary_path:
        vocabulary = set([])
        logger.info("Loading {} vocabulary from {}".format(tag,
                                                           vocabulary_path))

        if not os.path.exists(vocabulary_path):
            raise RuntimeError(
                "{} vocabulary not found at {}!".format(tag, vocabulary_path))
        else:
            with open(vocabulary_path) as f:
                for line in f:
                    if len(line.strip()) == 0:
                        continue
                    word = line.strip().split()[0]
                    vocabulary.add(word)
    return vocabulary


class OrderedIterator(torchtext.data.Iterator):
    """ Ordered Iterator Class """

    def create_batches(self):
        """ Create batches """
        if self.train:
            def _pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = _pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))
                

class AMRBatch:
    index = 0
    def __init__(self, data, dataset, device, train):
        self.batch_size = len(data)
        self.dataset = dataset
        self.indices = torch.LongTensor(self.batch_size)
        for i in range(self.batch_size):
            self.indices[i] = AMRBatch.index
            AMRBatch.index += 1
        if device != 'cpu':
            self.indices = self.indices.cuda()
        self.indices = Variable(self.indices)
        
        str_stoi = dataset.fields['src'].vocab.stoi
        if 'tgt' in dataset.fields:
            tgt_stoi = dataset.fields['tgt'].vocab.stoi
        src_length = 0
        for x in data:
            src_length = max(src_length, len(x.src))
        src_lengths = []
        tgt_length = 0
        srcs = []
        tgts = []
        alignments = []
        srcmaps = []
        supports = []
        for example in data:
            size = len(example.src)
            size_matrix = example.src_graph.matrix.size()
            src_lengths.append(size)
            # src_lengths.append(size_matrix[1])
            tree = read_tree(example.src_graph.parents)                
            adj = torch.LongTensor(size_matrix[0], src_length, src_length).zero_()
            adj[:, 0 : size_matrix[1], 0 : size_matrix[1]] = example.src_graph.matrix
            src_identifiers = [str_stoi["<blank>"]] * (src_length)
            for i, item in enumerate(example.src):
                src_identifiers[i] = str_stoi[item]
            example_src = (tree, Variable(adj), 
                         Variable(torch.from_numpy(np.asarray(src_identifiers)).long()))                    
            if device != 'cpu':
                example_src = (example_src[0], example_src[1], example_src[2].cuda())                         
            srcs.append(example_src[2])
            supports.append((example_src[0], example_src[1]))            
            if hasattr(example, 'src_map'):
                src_map = torch.FloatTensor(src_length, src_length + 2).zero_()
                for i in range(example.src_map.size(0)):
                    src_map[i][example.src_map[i]] = 1
                src_map = Variable(src_map)
                if device != 'cpu':
                    src_map = src_map.cuda()
                srcmaps.append(src_map)            
                
            if 'tgt' in dataset.fields:
                tgt_length = max(tgt_length, len(example.tgt))
        
        if 'tgt' in dataset.fields:
            for example in data:     
                tgt_identifiers = []
                for i, word in enumerate(example.tgt):
                    tgt_identifiers.append(tgt_stoi[word])
                tgt_identifiers = [tgt_stoi["<s>"]] + tgt_identifiers + [tgt_stoi["</s>"]]
                tgt_identifiers += [tgt_stoi["<blank>"]] * (tgt_length + 2 - len(tgt_identifiers))
                example_tgt = Variable(torch.from_numpy(np.asarray(tgt_identifiers)).long())
                if device != 'cpu':
                    example_tgt = example_tgt.cuda()
                tgts.append(example_tgt)

                if getattr(example, "alignment", None) is not None:
                    align_identifiers = []
                    for word in example.alignment:
                        align_identifiers.append(word)
                    align_identifiers += [0] * (tgt_length + 2 - len(align_identifiers))
                    example_alignment = Variable(torch.from_numpy(np.asarray(align_identifiers)).long())
                    if device != 'cpu':
                        example_alignment = example_alignment.cuda()
                    alignments.append(example_alignment)
        
        lengths = torch.from_numpy(np.asarray(src_lengths)).long()

        trees = []
        adjs = []
        for t, a in supports:
            trees.append(t)
            adjs.append(a)
        self.src = (torch.stack(srcs, 1), lengths)
        self.src_parents = trees
        self.src_adj = torch.stack(adjs, 1)
        # self.tgt = torch.stack(tgts, 1)
        
        if srcmaps != []:
            self.src_map = torch.stack(srcmaps, 1)
        if tgts != []:
            self.tgt = torch.stack(tgts, 1)
            if srcmaps != []:
                self.alignment = torch.stack(alignments, 1)        
        

class AMRIterator(torchtext.data.Iterator):
    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=lambda new, count, sofar: count, train=True, repeat=None, 
                 shuffle=None, sort=None, sort_within_batch=None): 
        super(AMRIterator, self).__init__(dataset=dataset, batch_size=batch_size, sort_key=sort_key, 
                device=device, batch_size_fn=batch_size_fn, train=train, repeat=repeat, 
                shuffle=shuffle, sort=sort, sort_within_batch=sort_within_batch)

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = AMRBatch(minibatch, self.dataset, self.device, self.train)
                yield batch
                
            if not self.repeat:
                raise StopIteration


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train, iterator_type):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train
        self.iterator_type = iterator_type

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset.examples = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        if self.iterator_type == 'amr':
            return AMRIterator(
                dataset=self.cur_dataset, batch_size=self.batch_size,
                batch_size_fn=self.batch_size_fn,
                device=self.device, train=self.is_train,
                sort=True, sort_within_batch=True,
                repeat=False)
        else:
            return OrderedIterator(
                dataset=self.cur_dataset, batch_size=self.batch_size,
                batch_size_fn=self.batch_size_fn,
                device=self.device, train=self.is_train,
                sort=False, sort_within_batch=True,
                repeat=False)
        


def build_dataset_iter(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over. We implement simple ordered iterator strategy here,
    but more sophisticated strategy like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    if is_train and opt.batch_type == "tokens":
        def batch_size_fn(new, count, sofar):
            """
            In token batching scheme, the number of sequences is limited
            such that the total number of src/tgt tokens (including padding)
            in a batch <= batch_size
            """
            # Maintains the longest src and tgt length in the current batch
            global max_src_in_batch, max_tgt_in_batch
            # Reset current longest length at a new batch (count=1)
            if count == 1:
                max_src_in_batch = 0
                max_tgt_in_batch = 0
            # Src: <bos> w1 ... wN <eos>
            max_src_in_batch = max(max_src_in_batch, len(new.src) + 2)
            # Tgt: w1 ... wN <eos>
            max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 1)
            src_elements = count * max_src_in_batch
            tgt_elements = count * max_tgt_in_batch
            return max(src_elements, tgt_elements)
    else:
        batch_size_fn = None

    if opt.gpu_ranks:
        device = "cuda"
    else:
        device = "cpu"

    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train, opt.data_type)


def lazily_load_dataset(corpus_type, opt):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def _load_fields(dataset, data_type, opt, checkpoint):
    if checkpoint is not None:
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = load_fields_from_vocab(
            checkpoint['vocab'], data_type)
    else:
        fields = load_fields_from_vocab(
            torch.load(opt.data + '.vocab.pt'), data_type)
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    if data_type == 'text' or data_type == 'amr':
        logger.info(' * vocabulary size. source = %d; target = %d' %
                    (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    else:
        logger.info(' * vocabulary size. target = %d' %
                    (len(fields['tgt'].vocab)))

    return fields


def _collect_report_features(fields):
    src_features = collect_features(fields, side='src')
    tgt_features = collect_features(fields, side='tgt')

    return src_features, tgt_features
