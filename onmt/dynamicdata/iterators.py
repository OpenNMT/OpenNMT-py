import collections
import itertools
import os
import re
import random
from glob import glob

from .utils import *

RE_SHARD = re.compile(r'([0-9]*)\.([a-z]*)(\.gz)?')

def infinite_iterator(iterator_factory):
    while True:
        yield from iterator_factory()

class GroupEpoch():
    def __init__(self, data_config, group):
        self.data_config = data_config
        self.group = group
        self.indices = set()
        self.files = collections.defaultdict(list)
        self.compressed = False
        self.epoch = 0
        self.groupdir = os.path.join(
            self.data_config['meta']['shard']['rootdir'],
            group)
        self.group_type = self.data_config['groups'][group]['type']
        if self.group_type == 'para':
            self.sides = ('src', 'tgt')
        elif self.group_type == 'mono':
            self.sides = ('mono',)
        else:
            raise Exception('Unrecognized group type "{}"'.format(
                self.group_type))
        self._find_shards()

    def _find_shards(self):
        for filename in os.listdir(self.groupdir):
            m = RE_SHARD.match(filename)
            if not m:
                continue
            index, suffix, ext = m.groups()
            self.indices.add(index)
            path = os.path.join(self.groupdir, filename)
            self.files[(suffix, index)] = path
            if ext == '.gz':
                self.compressed = True
            elif self.compressed:
                raise Exception('Cannot mix compressed and uncompressed')
        for side in self.sides:
            for index in self.indices:
                if not (side, index) in self.files:
                    raise Exception('Missing shard: {} {}, {}'.format(
                        self.group, side, index))

    def yield_epoch(self):
        self.epoch += 1
        all_shards = []
        for index in self.indices:
            all_shards.append([self.files[(side, index)] for side in self.sides])
        random.shuffle(all_shards)
        yield from all_shards

def debug(stream, prefix='debug'):
    for item in stream:
        print('{}: {}'.format(prefix, item))
        yield item

class ShardIterator():
    def __init__(self, group, files, transforms):
        self.group = group
        self.files = files
        self.transforms = transforms

    def __call__(self, is_train=True):
        fobjs = [open(path, 'r') for path in self.files]
        tokenized = [self.tokenize(stream) for stream in fobjs]
        transposed = self.transpose(tokenized)
        if is_train:
            random.shuffle(transposed)
        for fobj in fobjs:
            fobj.close()
        transformed = self.transform(transposed, is_train)
        #transformed = debug(transformed, 'transformed')
        indexed = self.add_index(transformed)
        yield from indexed

    def tokenize(self, stream):
        for line in stream:
            yield tuple(line.rstrip('\n').split())

    def transpose(self, streams):
        tpls = list(zip(*streams))
        return tpls

    def transform(self, stream, is_train):
        for tpl in stream:
            for transform in self.transforms:
                tpl = transform.apply(tpl, self.group, is_train)
            if tpl is None:
                # the last transform can filter by returning None
                if not is_train:
                    raise Exception('Cannot filter validation set')
                continue
            yield tpl

    def add_index(self, stream):
        for i, tpl in enumerate(stream):
            yield tpl + (i,)

def yield_infinite(group_epoch, group, transforms, is_train):
    for tpl in infinite_iterator(group_epoch.yield_epoch):
        si = ShardIterator(group, tpl, transforms)
        yield from si(is_train=is_train)

def yield_once(group_epoch, group, transforms, is_train):
    for tpl in group_epoch.yield_epoch():
        si = ShardIterator(group, tpl, transforms)
        yield from si(is_train=is_train)

def yield_translate(files, group, transforms):
    for tpl in files:
        si = ShardIterator(group, tpl, transforms)
        yield from si(is_train=False)

class TransformReader():
    def __init__(self, group, transforms):
        self.group = group
        self.transforms = transforms

    def read(self, src_file, side, _dir=None):
        if side != 'src':
            # TODO: make a hacky thing that buffers the joint data
            # and provides access to src and tgt separately via
            # two objects having a read method
            raise Exception("dynamicdata doesn't support tgt")
        print('src_file', src_file, type(src_file))
        files = [(src_file,)]
        stream = yield_translate(files, self.group, self.transforms)
        for (src, idx) in stream:
            yield {'src': src, 'indices': idx}

class MixingWeightSchedule():
    def __init__(self, data_config, keys):
        self.keys = keys
        self.schedule_steps = self._list(
            data_config['meta']['train'].get('mixing_weight_schedule', []))
        self.schedule_steps.append(None)
        self.mixing_weights = {
            key: self._list(data_config['groups'][key]['weight'])
            for key in keys}
        self.next_threshold = 0

    def _list(self, val):
        if isinstance(val, int):
            return [val]
        return list(val)

    def __call__(self, i):
        if self.next_threshold is None:
            # no adjustments left
            return None
        if i < self.next_threshold:
            # no adjustment yet
            return None
        self.next_threshold = self.schedule_steps.pop(0)
        new_weights = []
        for key in self.keys:
            new_weights.append(self.mixing_weights[key].pop(0))
        return new_weights

    def min_bucket_size(self):
        sums = []
        for tpl in zip(*self.mixing_weights.values()):
            sums.append(sum(tpl))
        return max(sums)

class GroupMixer():
    def __init__(self, data_config, group_streams, bucket_size=2048):
        self.data_config = data_config
        self.group_streams = group_streams
        self.keys = sorted(self.group_streams.keys())

        self.schedule = MixingWeightSchedule(data_config, self.keys)
        self.current_weights = None
        min_bucket_size = self.schedule.min_bucket_size()
        if bucket_size < min_bucket_size:
            bucket_size = min_bucket_size
            print('increased bucket_size to {}'.format(bucket_size))
        self.bucket_size = bucket_size

    def __call__(self):
        self.maybe_adjust_mix(0)
        mixed = self.mix()
        bucketed = self.bucket(mixed)
        yield from bucketed

    def mix(self):
        # using this while-try-next construction in order to allow
        # self.mixed to be replaced by mix weight schedule while looping
        while True:
            try:
                yield next(self.mixed)
            except StopIteration:
                break

    def bucket(self, stream):
        stream = iter(stream)
        while True:
            bucket = list(itertools.islice(stream, self.bucket_size))
            if len(bucket) == 0:
                break
            yield bucket

    def maybe_adjust_mix(self, i):
        new_weights = self.schedule(i)
        if new_weights is not None:
            self.current_weights = new_weights
            print('***', i, 'set weights to', self.current_weights)
        self.mixed = weighted_roundrobin(
            [self.group_streams[key] for key in self.keys],
            self.current_weights)


def build_mixer(data_config, transforms, is_train=True, bucket_size=1):
    split = 'train' if is_train else 'valid'
    group_epochs = {}
    group_streams = {}
    for group in data_config['groups']:
        if data_config['groups'][group]['split'] != split:
            continue
        group_epoch = GroupEpoch(data_config, group)
        if is_train:
            stream = yield_infinite(group_epoch, group, transforms[group], is_train=is_train)
        else:
            # yield validation data only once
            stream = yield_once(group_epoch, group, transforms[group], is_train=is_train)
        group_epochs[group] = group_epoch
        group_streams[group] = stream
    mixer = GroupMixer(data_config, group_streams, bucket_size=bucket_size)
    return mixer, group_epochs 
