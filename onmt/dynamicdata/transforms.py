import collections
import functools
import gzip
import itertools
import math
import numpy as np
import onmt.inputters
import os
import random
import torchtext

from scipy.special import softmax

from onmt.utils.logging import logger
from .vocab import Vocabulary
from .utils import UNDER

WARM_UP = 50000


class TransformModel():
    """ A model from which individual Transforms
    can be instantiated for each task """
    def __init__(self, data_config):
        self.data_config = data_config

    def warm_up(self, vocabs):
        pass

    def get_transform(self, transform, task):
        raise NotImplementedError()


class Transform():
    """ A preprocessing step, reapplied separately
    each time a minibatch is instantiated. """
    def __init__(self, data_config):
        pass

    def warm_up(self, vocabs):
        pass

    def set_train_opts(self, data_config):
        pass

    def get_specials(self):
        """ returns special tokens added by this transform """
        return set()

    def apply(self, tpl, task, is_train=True):
        return tpl

    def stats(self):
        return ()

    def __repr__(self):
        return self.__class__.__name__


class SimpleTransform(Transform):
    """ A Transform that is its own TransformModel """
    def __init__(self, data_config):
        pass

    def get_transform(self, transform, task):
        return self


class DuplicateMonoTransform(SimpleTransform):
    def apply(self, tpl, task, is_train=True):
        src = tuple(tpl[0])
        return src, src


class ReorderTransform(SimpleTransform):
    def __init__(self, data_config):
        super().__init__(data_config)

    def set_train_opts(self, data_config):
        self.max_dist = data_config['meta']['train'].get(
            'reorder_max_dist', 3)

    def _perturb(self, tokens):
        indices = [(i + random.uniform(0, self.max_dist), token)
                   for (i, token) in enumerate(tokens)]
        indices.sort()
        return tuple(token for (i, token) in indices)

    def apply(self, tpl, task, is_train=True):
        if not is_train:
            return tpl
        src, tgt = tpl
        src = self._perturb(src)
        return src, tgt

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.max_dist)


class DropTransform(SimpleTransform):
    def __init__(self, data_config):
        super().__init__(data_config)
        self._sum_draw = 0
        self._sum_toks = 0

    def set_train_opts(self, data_config):
        self.temperature = data_config['meta']['train'].get(
            'drop_temperature', 2.0) * -1

    def _drop(self, tokens):
        n_tokens = len(tokens)
        indices = np.arange(n_tokens)
        logits = indices * self.temperature
        logits = softmax(logits)
        n_drops = np.random.choice(indices, p=logits)
        self._sum_draw += n_drops
        self._sum_toks += n_tokens

        drop_indices = set(np.random.choice(indices,
                                            size=n_drops,
                                            replace=False))
        tokens = [tok for (i, tok) in enumerate(tokens)
                  if i not in drop_indices]
        return tokens

    def apply(self, tpl, task, is_train=True):
        if not is_train:
            return tpl
        src, tgt = tpl
        src = self._drop(src)
        return src, tgt

    def stats(self):
        if self._sum_toks == 0:
            yield('no tokens dropped')
        else:
            yield('tokens dropped {} / {} = {}'.format(
                self._sum_draw,
                self._sum_toks,
                self._sum_draw / self._sum_toks))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.temperature)


class SwitchOutTransformModel(TransformModel):
    def __init__(self, data_config):
        super().__init__(data_config)
        self.vocab = None

    def warm_up(self, vocabs):
        # assumes shared vocab
        self.vocab = list(vocabs['shared'].keys())

    def get_transform(self, transform, task):
        return SwitchOutTransform(self.data_config, self.vocab)


class SwitchOutTransform(Transform):
    def __init__(self, data_config, vocab):
        super().__init__(data_config)
        self.vocab = vocab
        self._sum_draw = 0
        self._sum_toks = 0

    def set_train_opts(self, data_config):
        self.temperature = data_config['meta']['train'].get(
            'switchout_temperature', 2.0) * -1

    def _replace(self, token):
        return random.choice(self.vocab)

    def _switchout(self, tokens):
        n_tokens = len(tokens)
        indices = np.arange(n_tokens)
        logits = indices * self.temperature
        logits = softmax(logits)
        n_switchouts = np.random.choice(indices, p=logits)
        self._sum_draw += n_switchouts
        self._sum_toks += n_tokens

        switchout_indices = set(np.random.choice(indices,
                                                 size=n_switchouts,
                                                 replace=False))
        out = []
        for (i, tok) in enumerate(tokens):
            if i in switchout_indices:
                out.append(self._replace(tok))
            else:
                out.append(tok)
        return tuple(out)

    def apply(self, tpl, task, is_train=True):
        if not is_train:
            return tpl
        src, tgt = tpl
        src = self._switchout(src)
        tgt = self._switchout(tgt)
        return src, tgt

    def stats(self):
        if self._sum_toks == 0:
            yield('no tokens switched out')
        else:
            yield('switchout  {} / {} = {}'.format(
                self._sum_draw,
                self._sum_toks,
                self._sum_draw / self._sum_toks))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.temperature)


class WbNoiseTransformModel(TransformModel):
    def __init__(self, data_config):
        super().__init__(data_config)
        self.vocab = None

    def warm_up(self, vocabs):
        # assumes shared vocab
        self.vocab = set(vocabs['shared'].keys())

    def get_transform(self, transform, task):
        return WbNoiseTransform(self.data_config, self.vocab)


class WbNoiseTransform(SwitchOutTransform):
    def set_train_opts(self, data_config):
        self.temperature = data_config['meta']['train'].get(
            'wb_noise_temperature', 2.0) * -1

    def _replace(self, token):
        if token[0] == UNDER:
            # wb -> no wb
            proposed = token[1:]
        else:
            # no wb -> wb
            proposed = UNDER + token
        if len(proposed) == 0:
            return token
        if proposed not in self.vocab:
            return token
        return proposed

    def stats(self):
        if self._sum_toks == 0:
            yield('no wb_noise')
        else:
            yield('wb_noise  {} / {} = {}'.format(
                self._sum_draw,
                self._sum_toks,
                self._sum_draw / self._sum_toks))


class InsertionTransformModel(TransformModel):
    def __init__(self, data_config):
        super().__init__(data_config)
        self.vocab = None

    def warm_up(self, vocabs):
        # assumes shared vocab
        self.vocab = list(vocabs['shared'].keys())

    def get_transform(self, transform, task):
        return InsertionTransform(self.data_config, self.vocab)


class InsertionTransform(SwitchOutTransform):
    def set_train_opts(self, data_config):
        self.temperature = data_config['meta']['train'].get(
            'insertion_temperature', 2.0) * -1

    def _switchout(self, tokens):
        n_tokens = len(tokens)
        indices = np.arange(n_tokens)
        logits = indices * self.temperature
        logits = softmax(logits)
        n_switchouts = np.random.choice(indices, p=logits)
        self._sum_toks += n_tokens

        switchout_indices = set(np.random.choice(indices,
                                                 size=n_switchouts,
                                                 replace=False))
        out = []
        for (i, tok) in enumerate(tokens):
            if i in switchout_indices:
                # first the insertion
                self._sum_draw += 1
                out.append(self._replace(tok))
            # then the old token
            out.append(tok)
        return tuple(out)

    def stats(self):
        if self._sum_toks == 0:
            yield('no insertion')
        else:
            yield('insertion  {} / {} = {}'.format(
                self._sum_draw,
                self._sum_toks,
                self._sum_draw / self._sum_toks))


class FilterTooLongTransform(SimpleTransform):
    def __init__(self, data_config):
        super().__init__(data_config)
        self._n_dropped = 0
        self._n_kept = 0

    def set_train_opts(self, data_config):
        self.max_len = data_config['meta']['train']['max_len']

    def apply(self, tpl, task, is_train=True):
        if not is_train:
            return tpl
        if any(len(side) > self.max_len for side in tpl):
            self._n_dropped += 1
            return None
        self._n_kept += 1
        return tpl

    def stats(self):
        tot = self._n_dropped + self._n_kept
        if tot <= 0:
            yield('nothing filtered')
        else:
            yield('length filtered {} / {} = {}'.format(
                self._n_dropped, tot, self._n_dropped / tot))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.max_len)


class PrefixTransformModel(TransformModel):
    def __init__(self, data_config):
        super().__init__(data_config)

    def get_transform(self, transform, task):
        # FIXME can NOT override in set_train_opts
        src_lang = self.data_config['tasks'][task]['meta']['src_lang']
        tgt_lang = self.data_config['tasks'][task]['meta']['tgt_lang']
        prefix = ('<FROM_{}>'.format(src_lang),
                  '<TO_{}>'.format(tgt_lang))
        try:
            extra_prefix = \
                self.data_config['tasks'][task]['meta']['extra_prefix']
            prefix = prefix + (extra_prefix,)
        except KeyError:
            pass
        return PrefixTransform(prefix)


class PrefixTransform(Transform):
    def __init__(self, prefix):
        self.prefix = tuple(prefix)

    def get_specials(self):
        return set(self.prefix)

    def apply(self, tpl, task, is_train=True):
        # src is required, tgt is optional
        src, tail = tpl[0], tpl[1:]
        src = self.prefix + tuple(src)
        return (src,) + tail

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.prefix)


def mix_half(xs, ys):
    assert len(xs) == len(ys)
    len_seq = len(xs)
    indices = list(range(len_seq))
    random.shuffle(indices)
    chosen = indices[:(len_seq // 2)]
    out_a = [x if i in chosen else y for (i, (x, y)) in enumerate(zip(xs, ys))]
    out_b = [y if i in chosen else x for (i, (x, y)) in enumerate(zip(xs, ys))]
    return out_a, out_b


def flatten(deep):
    out = []
    for token in deep:
        out.extend(token)
    return tuple(out)


def load_word_counts(data_config):
    vocab = Vocabulary(data_config)
    path = vocab.path('shared', segmentation='words')
    return vocab.load(path)


class MorfessorEmStdTransform(Transform):
    def __init__(self, data_config, seg_model, task):
        super().__init__(data_config)
        self.data_config = data_config
        self.seg_model = seg_model
        self.task = task
        # can NOT override in set_train_opts
        self.n_samples = data_config['meta']['train'].get(
            'seg_n_samples', 5)
        self.theta = data_config['meta']['train'].get(
            'seg_theta', 0.5)
        print('n_samples {} theta {}'.format(self.n_samples, self.theta))
        self._cache = StdSampleCache(
            self.seg_model,
            n_samples=self.n_samples,
            addcount=0,
            theta=self.theta,
            maxlen=30)

    def warm_up(self, vocabs):
        # load the word (not subword) counts
        counts = load_word_counts(self.data_config)
        # populate cache for most frequent words
        for w, c in counts.most_common(WARM_UP):
            self._cache._populate_cache(w)

    def apply(self, tpl, task, is_train=True):
        out = []
        for tokens in tpl:
            morphs = []
            for token in tokens:
                morphs.extend(self._cache.segment(token))
            out.append(tuple(morphs))
        return tuple(out)

    def stats(self):
        yield('hits {} vs misses {}'.format(self._cache.hits,
                                            self._cache.misses))

    def __repr__(self):
        try:
            return '{}({}, {})'.format(self.__class__.__name__,
                                       self.n_samples,
                                       self.theta)
        except AttributeError:
            return '{}[{}]'.format(self.__class__.__name__, 'old')


class MorfessorEmTabooTransform(Transform):
    def __init__(self, data_config, seg_model, task):
        super().__init__(data_config)
        self.data_config = data_config
        self.seg_model = seg_model
        self.task = task
        self.lang = self.data_config['tasks'][task]['meta']['src_lang']
        self.prefix = ('<TABOO_AE_{}>'.format(self.lang),)
        # can NOT override in set_train_opts
        self.n_samples = data_config['meta']['train'].get(
            'seg_n_samples', 5)
        self.theta = data_config['meta']['train'].get(
            'seg_theta', 0.5)
        print('n_samples {} theta {}'.format(self.n_samples, self.theta))
        self._cache = TabooSampleCache(
            self.seg_model,
            n_samples=self.n_samples,
            addcount=0,
            theta=self.theta,
            maxlen=30)

    def warm_up(self, vocabs):
        # load the word (not subword) counts
        counts = load_word_counts(self.data_config)
        # populate cache for most frequent words
        for w, c in counts.most_common(WARM_UP):
            self._cache._populate_cache(w)

    def get_specials(self):
        return set(self.prefix)

    def apply(self, tpl, task, is_train=True):
        # src -> xs, ys
        src, = tpl
        xs, ys = zip(*[self._cache.segment(token) for token in src])
        # xs, ys -mix-> src, tgt
        src, tgt = mix_half(xs, ys)
        src = flatten(src)
        tgt = flatten(tgt)
        # prepend '<TABOO_AE_{lang}>'
        src = self.prefix + src
        return src, tgt

    def stats(self):
        yield('hits {} vs misses {}'.format(self._cache.hits,
                                            self._cache.misses))

    def __repr__(self):
        try:
            return '{}({}, {})'.format(self.__class__.__name__,
                                       self.n_samples,
                                       self.theta)
        except AttributeError:
            return '{}[{}]'.format(self.__class__.__name__, 'old')


class MorfessorEmTabooReorderedTransform(MorfessorEmTabooTransform):
    """ taboo segmentation, after reordering src """
    def apply(self, tpl, task, is_train=True):
        # src -> xs, ys
        src, tgt = tpl
        all_morphs = sorted(set(src).union(tgt))
        xs, ys = zip(*[self._cache.segment(token) for token in all_morphs])
        # mix sides (per type)
        src_segs, tgt_segs = mix_half(xs, ys)
        # apply side-specific segmentation maps
        src_map = {morph: src_seg
                   for morph, src_seg
                   in zip(all_morphs, src_segs)}
        tgt_map = {morph: tgt_seg
                   for morph, tgt_seg
                   in zip(all_morphs, tgt_segs)}
        src = [src_map[token] for token in src]
        tgt = [tgt_map[token] for token in tgt]
        src = flatten(src)
        tgt = flatten(tgt)
        # prepend '<TABOO_AE_{lang}>'
        src = self.prefix + src
        return src, tgt


class MorfessorEmTransformModel():
    transform_classes = {
        'morfessor_em': MorfessorEmStdTransform,
        'morfessor_em_taboo': MorfessorEmTabooTransform,
        'morfessor_em_taboo_reordered': MorfessorEmTabooReorderedTransform,
    }

    def __init__(self, data_config):
        self.data_config = data_config

    def warm_up(self, vocabs):
        import morfessor
        # load the segmentation model
        io = morfessor.MorfessorIO()
        expected = io.read_expected_file(
            self.data_config['meta']['train']['segmentation_model'])
        self.seg_model = morfessor.BaselineModel(
            use_em=True,
            em_substr=expected)

    def get_transform(self, transform, task):
        try:
            transform_cls = self.transform_classes[transform]
        except KeyError:
            raise Exception('Unknown transform {}'.format(transform))
        return transform_cls(self.data_config, self.seg_model, task)


class SentencepieceTransform(SimpleTransform):
    def __init__(self, data_config):
        super().__init__(data_config)
        self.data_config = data_config
        self.model_path = \
            self.data_config['meta']['train']['segmentation_model']
        # can NOT override in set_train_opts
        self.n_samples = data_config['meta']['train'].get(
            'seg_n_samples', -1)
        self.theta = data_config['meta']['train'].get(
            'seg_theta', 0.5)
        if data_config['meta']['shard'].get('pretokenize', False):
            raise Exception(
                'SentencepieceTransform should not be used with "pretokenize"')
        if not data_config['meta']['shard'].get('predetokenize', False):
            logger.warn(
                'SentencepieceTransform used without "predetokenize". '
                'Make sure that the input is not tokenized.')

    def warm_up(self, vocabs=None):
        # load the segmentation model
        import sentencepiece as spm
        self.seg_model = spm.SentencePieceProcessor()
        self.seg_model.Load(self.model_path)

    def apply(self, tpl, task, is_train=True):
        out = []
        for tokens in tpl:
            out.append(tuple(self.seg_model.SampleEncodeAsPieces(
                ' '.join(tokens), self.n_samples, self.theta)))
        return tuple(out)

    def __getstate__(self):
        return {'data_config': self.data_config,
                'model_path': self.model_path,
                'n_samples': self.n_samples,
                'theta': self.theta}

    def __setstate__(self, d):
        self.data_config = d['data_config']
        self.model_path = d['model_path']
        self.n_samples = d['n_samples']
        self.theta = d['theta']
        self.warm_up()


class DeterministicSegmentationTransform(SimpleTransform):
    def __init__(self, data_config):
        super().__init__(data_config)
        self.data_config = data_config
        self.mapping = None
        self.mapping_path = \
            self.data_config['meta']['train']['segmentation_model']
        n_samples = data_config['meta']['train'].get(
            'seg_n_samples', 1)
        if n_samples != 1:
            raise Exception(
                'DeterministicSegmentationTransform requires seg_n_samples=1,'
                ' not {}'.format(n_samples))

    def warm_up(self, vocabs=None):
        # load the word to segmentation mapping
        # format: "<word> <tab> <morph1> <space> ... <morphN>"
        self.mapping = {}
        with open(self.mapping_path, 'r') as lines:
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    continue
                word, seg = line.split(None, 1)
                seg = tuple(seg.split(' '))
                self.mapping[word] = seg

    def apply(self, tpl, task, is_train=True):
        out = []
        for tokens in tpl:
            mapped = []
            for token in tokens:
                # intentionally die if token not in mapping
                mapped.extend(self.mapping[token])
            out.append(tuple(mapped))
        return tuple(out)


class SampleCache(object):
    def __init__(self, model, n_samples=5,
                 addcount=0, theta=0.5, maxlen=30):
        import morfessor
        self.model = model
        self.n_samples = n_samples
        self.addcount = addcount
        self.theta = theta
        self.maxlen = maxlen
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self._categorical = morfessor.utils.categorical

    def segment(self, compound):
        if compound not in self.cache:
            self.misses += 1
            self._populate_cache(compound)
        else:
            self.hits += 1
        analyses, distr = self.cache[compound]
        if len(analyses) == 1:
            return analyses[0]
        _, sample = self._categorical(analyses, distr)
        return sample


class StdSampleCache(SampleCache):
    def _populate_cache(self, compound):
        n_best = self.model.viterbi_nbest(
            compound,
            self.n_samples,
            addcount=self.addcount,
            theta=self.theta,
            maxlen=self.maxlen)
        analyses, logps = zip(*n_best)
        distr = [math.exp(-x) for x in logps]
        divisor = sum(distr)
        if divisor == 0:
            distr = [1]
            analyses = [analyses[0]]
        else:
            distr = [x / divisor for x in distr]
        self.cache[compound] = (analyses, distr)


class TabooSampleCache(SampleCache):
    def _populate_cache(self, compound):
        samples = []
        logps = []
        # first side
        n_best = self.model.viterbi_nbest(
            compound,
            self.n_samples,
            addcount=self.addcount,
            theta=self.theta,
            maxlen=self.maxlen)

        for first, first_logp in n_best:
            taboo = [morph for morph in first if len(morph) > 1]
            second, second_logp = self.model.sample_segment(
                compound, theta=self.theta, maxlen=self.maxlen, taboo=taboo)
            logp = first_logp + second_logp
            samples.append((first, second))
            logps.append(logp)

        distr = [math.exp(-x) for x in logps]
        divisor = sum(distr)
        if divisor == 0:
            distr = [1]
            samples = [samples[0]]
        else:
            distr = [x / divisor for x in distr]
        self.cache[compound] = (samples, distr)


DEFAULT_TRANSFORMS = {
    'duplicate_mono': DuplicateMonoTransform,
    'reorder': ReorderTransform,
    'drop': DropTransform,
    'switchout': SwitchOutTransformModel,
    'wb_noise': WbNoiseTransformModel,
    'insertion': InsertionTransformModel,
    'lang_prefix_both': PrefixTransformModel,
    'morfessor_em': MorfessorEmTransformModel,
    'morfessor_em_taboo': MorfessorEmTransformModel,
    'morfessor_em_taboo_reordered': MorfessorEmTransformModel,
    'sentencepiece': SentencepieceTransform,
    'deterministic_segmentation': DeterministicSegmentationTransform,
    'filter_too_long': FilterTooLongTransform,
}


def make_transform_models(data_config, vocabs, custom_transforms=None):
    transform_model_classes = dict(DEFAULT_TRANSFORMS)
    if custom_transforms is not None:
        transform_model_classes.update(custom_transforms)
    transform_models = {}
    for key in data_config['_transforms']:
        transform_model = transform_model_classes[key](data_config)
        transform_model.warm_up(vocabs)
        transform_models[key] = transform_model
    return transform_models


def make_transforms(transform_models, data_config, vocabs):
    transforms = {}
    for task in data_config['tasks']:
        keys = data_config['tasks'][task]['transforms']
        transforms[task] = [transform_models[key].get_transform(key, task)
                            for key in keys]
        for transform in transforms[task]:
            transform.warm_up(vocabs)
    return transforms


def get_specials(transforms):
    # assumes shared vocabulary
    all_specials = set()
    for task_transforms in transforms.values():
        for transform in task_transforms:
            all_specials.update(transform.get_specials())
    return all_specials


def set_train_opts(data_config, transforms):
    for task_transforms in transforms.values():
        for transform in task_transforms:
            transform.set_train_opts(data_config)
