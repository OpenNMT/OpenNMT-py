import collections
import functools
import gzip
import itertools
import math
import morfessor
import numpy as np
import onmt.inputters
import os
import random
import torchtext

from scipy.special import softmax

from .vocab import Vocabulary

WARM_UP = 50000

class TransformModel():
    """ A model from which individual Transforms
    can be instantiated for each group """
    def __init__(self, data_config):
        self.data_config = data_config

    def warm_up(self):
        pass

    def get_transform(self, transform, group):
        raise NotImplementedError()

class Transform():
    """ A preprocessing step, reapplied separately
    each time a minibatch is instantiated. """
    def __init__(self, data_config):
        pass

    def warm_up(self):
        pass

    def set_train_opts(self, data_config):
        pass

    def get_specials(self):
        """ returns special tokens added by this transform """
        return set()

    def apply(self, tpl, group, is_train=True):
        return tpl

    def stats(self):
        pass

class SimpleTransform(Transform):
    """ A Transform that is its own TransformModel """
    def __init__(self, data_config):
        pass

    def get_transform(self, transform, group):
        return self

class DuplicateMonoTransform(SimpleTransform):
    def apply(self, tpl, group, is_train=True):
        src = tuple(tpl[0])
        return src, src

class PeturbOrderTransform(SimpleTransform):
    def __init__(self, data_config):
        super().__init__(data_config)

    def set_train_opts(self, data_config):
        self.max_dist = data_config['meta']['train'].get(
            'peturb_order_max_dist', 3)

    def _peturb(self, tokens):
        indices = [(i + random.uniform(0, self.max_dist), token)
                   for (i, token) in enumerate(tokens)]
        indices.sort()
        return tuple(token for (i, token) in indices)

    def apply(self, tpl, group, is_train=True):
        if not is_train:
            return tpl
        src, tgt = tpl
        src = self._peturb(src)
        return src, tgt

class DropTransform(SimpleTransform):
    def __init__(self, data_config):
        super().__init__(data_config)
        self._sum_draw = 0
        self._sum_toks = 0

    def set_train_opts(self, data_config):
        self.temperature = data_config['meta']['train'].get(
            'drop_temperature', 1.3) * -1

    def _drop(self, tokens):
        n_tokens = len(tokens)
        indices = np.arange(n_tokens)
        logits = indices * self.temperature
        logits = softmax(logits)
        n_drops = np.random.choice(indices, p=logits)
        self._sum_draw += n_drops
        self._sum_toks += n_tokens

        drop_indices = set(np.random.choice(indices, size=n_drops, replace=False))
        tokens = [tok for (i, tok) in enumerate(tokens) if i not in drop_indices]
        return tokens

    def apply(self, tpl, group, is_train=True):
        if not is_train:
            return tpl
        src, tgt = tpl
        src = self._drop(src)
        return src, tgt

    def stats(self):
        if self._sum_toks == 0:
            print('no tokens dropped')
        else:
            print('tokens dropped {} / {} = {}'.format(
                self._sum_draw, self._sum_toks, self._sum_draw / self._sum_toks))

class FilterTooLongTransform(SimpleTransform):
    def __init__(self, data_config):
        super().__init__(data_config)
        self._n_dropped = 0
        self._n_kept = 0

    def set_train_opts(self, data_config):
        self.max_len = data_config['meta']['train']['max_len']

    def apply(self, tpl, group, is_train=True):
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
            print('nothing filtered')
        else:
            print('length filtered {} / {} = {}'.format(
                self._n_dropped, tot, self._n_dropped / tot))

class PrefixTransformModel(TransformModel):
    def __init__(self, data_config):
        super().__init__(data_config)

    def get_transform(self, transform, group):
        # TODO could also implement other types of prefix
        src_lang = self.data_config['groups'][group]['meta']['src_lang']
        tgt_lang = self.data_config['groups'][group]['meta']['tgt_lang']
        prefix = ('<FROM_{}>'.format(src_lang),
                  '<TO_{}>'.format(tgt_lang))
        return PrefixTransform(prefix)

class PrefixTransform(Transform):
    def __init__(self, prefix):
        self.prefix = tuple(prefix)

    def get_specials(self):
        return set(self.prefix)

    def apply(self, tpl, group, is_train=True):
        # src is required, tgt is optional
        src, tail = tpl[0], tpl[1:]
        src = self.prefix + tuple(src)
        return (src,) + tail


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
    def __init__(self, data_config, seg_model, group):
        super().__init__(data_config)
        self.data_config = data_config
        self.seg_model = seg_model
        self.group = group
        # TODO params from config? can't override in set_train_opts
        self._cache = StdSampleCache(
            self.seg_model,
            n_samples=5,
            addcount=0,
            theta=0.5,
            maxlen=30)

    def warm_up(self):
        # load the word (not subword) counts
        counts = load_word_counts(self.data_config)
        # populate cache for most frequent words
        for w, c in counts.most_common(WARM_UP):
            self._cache._populate_cache(w)

    def apply(self, tpl, group, is_train=True):
        out = []
        for tokens in tpl:
            morphs = []
            for token in tokens:
                morphs.extend(self._cache.segment(token))
            out.append(tuple(morphs))
        return tuple(out)

    def stats(self):
        print('hits {} vs misses {}'.format(self._cache.hits, self._cache.misses))

class MorfessorEmTabooTransform(Transform):
    def __init__(self, data_config, seg_model, group):
        super().__init__(data_config)
        self.data_config = data_config
        self.seg_model = seg_model
        self.group = group
        self.lang = self.data_config['groups'][group]['meta']['src_lang']
        self.prefix = ('<TABOO_AE_{}>'.format(self.lang),)
        self._cache = TabooSampleCache(
            self.seg_model,
            n_samples=5,
            addcount=0,
            theta=0.5,
            maxlen=30)

    def warm_up(self):
        # load the word (not subword) counts
        counts = load_word_counts(self.data_config)
        # populate cache for most frequent words
        for w, c in counts.most_common(WARM_UP):
            self._cache._populate_cache(w)

    def get_specials(self):
        return set(self.prefix)

    def apply(self, tpl, group, is_train=True):
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
        print('hits {} vs misses {}'.format(self._cache.hits, self._cache.misses))

class MorfessorEmTransformModel():
    transform_classes = {
        'morfessor_em': MorfessorEmStdTransform,
        'morfessor_em_taboo': MorfessorEmTabooTransform,
    }

    def __init__(self, data_config):
        self.data_config = data_config

    def warm_up(self):
        # load the segmentation model
        io = morfessor.MorfessorIO()
        expected = io.read_expected_file(
            self.data_config['meta']['train']['segmentation_model'])
        self.seg_model = morfessor.BaselineModel(
            use_em=True,
            em_substr=expected)

    def get_transform(self, transform, group):
        try:
            transform_cls = self.transform_classes[transform]
        except KeyError:
            raise Exception('Unknown transform {}'.format(transform))
        return transform_cls(self.data_config, self.seg_model, group)

class SampleCache(object):
    def __init__(self, model, n_samples=5,
                 addcount=0, theta=0.5, maxlen=30):
            self.model = model
            self.n_samples = n_samples
            self.addcount = addcount
            self.theta = theta
            self.maxlen = maxlen
            self.cache = {}
            self.hits = 0
            self.misses = 0

    def segment(self, compound):
        if compound not in self.cache:
            self.misses += 1
            self._populate_cache(compound)
        else:
            self.hits += 1
        analyses, distr = self.cache[compound]
        #for a, d in zip(analyses, distr):
        #    print(a, d)
        if len(analyses) == 1:
            return analyses[0]
        _, sample = morfessor.utils.categorical(analyses, distr)
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
            #print('first', first_logp, 'second', second_logp)
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
    'peturb_order': PeturbOrderTransform,
    'drop': DropTransform,
    'lang_prefix_both': PrefixTransformModel,
    'morfessor_em': MorfessorEmTransformModel,
    'morfessor_em_taboo': MorfessorEmTransformModel,
    'filter_too_long': FilterTooLongTransform,
}

def make_transform_models(data_config, custom_transforms=None):
    transform_model_classes = dict(DEFAULT_TRANSFORMS)
    if custom_transforms is not None:
        transform_model_classes.update(custom_transforms)
    transform_models = {}
    for key in data_config['_transforms']:
        transform_model = transform_model_classes[key](data_config)
        transform_model.warm_up()
        transform_models[key] = transform_model
    return transform_models

def make_transforms(transform_models, data_config):
    transforms = {}
    for group in data_config['groups']:
        keys = data_config['groups'][group]['transforms']
        transforms[group] = [transform_models[key].get_transform(key, group)
                             for key in keys]
        for transform in transforms[group]:
            transform.warm_up()
    return transforms

def get_specials(transforms):
    # assumes shared vocabulary
    all_specials = set()
    for group_transforms in transforms.values():
        for transform in group_transforms:
            all_specials.update(transform.get_specials())
    return all_specials

def set_train_opts(data_config, transforms):
    for group_transforms in transforms.values():
        for transform in group_transforms:
            transform.set_train_opts(data_config)
