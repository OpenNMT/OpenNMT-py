import collections
import itertools
import os
import re
import random
from glob import glob

from .utils import *

from onmt.utils.logging import logger

RE_SHARD = re.compile(r'([0-9]*)\.([a-z]*)(\.gz)?')


def infinite_iterator(iterator_factory):
    while True:
        yield from iterator_factory()


class TaskEpoch():
    def __init__(self, data_config, task):
        self.data_config = data_config
        self.task = task
        self.indices = set()
        self.files = collections.defaultdict(list)
        self.compressed = False
        self.epoch = 0
        # share_inputs causes this task to use another one's data
        data_task = self.data_config['tasks'][task].get('share_inputs', task)
        self.taskdir = os.path.join(
            self.data_config['meta']['shard']['rootdir'],
            data_task)
        self.task_type = self.data_config['tasks'][task]['type']
        if self.task_type == 'para':
            self.sides = ('src', 'tgt')
        elif self.task_type == 'mono':
            self.sides = ('mono',)
        else:
            raise Exception('Unrecognized task type "{}"'.format(
                self.task_type))
        # reverse causes roles of src and tgt data to be reversed
        # useful for backtranslation
        # (note that meta src_lang and trg_lang are not modified)
        if self.data_config['meta']['train'].get('reverse', False):
            self.sides = self.sides[::-1]
        self._find_shards()
        logger.info('Shard task "{}" from "{}"'.format(self.task,
                                                       self.taskdir))

    def _find_shards(self):
        for filename in os.listdir(self.taskdir):
            m = RE_SHARD.match(filename)
            if not m:
                continue
            index, suffix, ext = m.groups()
            self.indices.add(index)
            path = os.path.join(self.taskdir, filename)
            self.files[(suffix, index)] = path
            if ext == '.gz':
                self.compressed = True
            elif self.compressed:
                raise Exception('Cannot mix compressed and uncompressed')
        for side in self.sides:
            for index in self.indices:
                if not (side, index) in self.files:
                    raise Exception('Missing shard: {} {}, {}'.format(
                        self.task, side, index))

    def yield_epoch(self):
        self.epoch += 1
        all_shards = []
        for index in self.indices:
            all_shards.append([self.files[(side, index)]
                               for side in self.sides])
        random.shuffle(all_shards)
        yield from all_shards


def debug(stream, prefix='debug'):
    for item in stream:
        print('{}: {}'.format(prefix, item))
        yield item


class ShardIterator():
    def __init__(self, task, files, transforms):
        self.task = task
        self.files = files
        self.transforms = transforms

    def tokenize(self, stream):
        for line in stream:
            yield tuple(line.rstrip('\n').split())

    def transpose(self, streams):
        tpls = list(zip(*streams))
        return tpls

    def transform(self, stream, is_train):
        for tpl in stream:
            for transform in self.transforms:
                tpl = transform.apply(tpl, self.task, is_train)
            if tpl is None:
                # the last transform can filter by returning None
                if not is_train:
                    raise Exception('Cannot filter validation set')
                continue
            yield tpl

    def add_index(self, stream):
        for i, tpl in enumerate(stream):
            yield tpl + (i,)


class TrainShardIterator(ShardIterator):
    def __call__(self, is_train=True):
        fobjs = [open(path, 'r') for path in self.files]
        tokenized = [self.tokenize(stream) for stream in fobjs]
        transposed = self.transpose(tokenized)
        if is_train:
            random.shuffle(transposed)
        for fobj in fobjs:
            fobj.close()
        # transposed = debug(transposed, 'transposed')
        transformed = self.transform(transposed, is_train)
        # transformed = debug(transformed, 'transformed')
        indexed = self.add_index(transformed)
        yield from indexed


class TranslateShardIterator(ShardIterator):
    def __call__(self, is_train=False):
        assert not is_train
        decoded = [self.decode(stream) for stream in self.files]
        tokenized = [self.tokenize(stream) for stream in decoded]
        transposed = self.transpose(tokenized)
        transformed = self.transform(transposed, is_train)
        # transformed = debug(transformed, 'transformed')
        indexed = self.add_index(transformed)
        yield from indexed

    def decode(self, stream):
        for bstr in stream:
            yield bstr.decode("utf-8")


def yield_infinite(task_epoch, task, transforms, is_train):
    for tpl in infinite_iterator(task_epoch.yield_epoch):
        si = TrainShardIterator(task, tpl, transforms)
        yield from si(is_train=is_train)


def yield_once(task_epoch, task, transforms, is_train):
    for tpl in task_epoch.yield_epoch():
        si = TrainShardIterator(task, tpl, transforms)
        yield from si(is_train=is_train)


def yield_translate(files, task, transforms):
    for tpl in files:
        si = TranslateShardIterator(task, tpl, transforms)
        yield from si(is_train=False)


def yield_debug(files, task, transforms, is_train):
    for tpl in zip(files):
        si = TrainShardIterator(task, tpl, transforms)
        yield from si(is_train=is_train)


class TransformReader():
    def __init__(self, task, transforms):
        self.task = task
        self.transforms = transforms

    def read(self, src_file, side, _dir=None):
        if side != 'src':
            # TODO: make a hacky thing that buffers the joint data
            # and provides access to src and tgt separately via
            # two objects having a read method
            raise Exception("dynamicdata doesn't support tgt")
        files = [(src_file,)]
        stream = yield_translate(files, self.task, self.transforms)
        for (src, idx) in stream:
            yield {'src': src, 'indices': idx}


class MixingWeightSchedule():
    def __init__(self, data_config, keys):
        self.keys = keys
        self.schedule_steps = self._list(
            data_config['meta']['train'].get('mixing_weight_schedule', []))
        self.schedule_steps.append(None)
        self.mixing_weights = {
            key: self._list(data_config['tasks'][key]['weight'],
                            len(self.schedule_steps))
            for key in keys}
        for key in keys:
            if len(self.mixing_weights[key]) != len(self.schedule_steps):
                raise Exception('task "{}" has {} mixing weights,'
                                'expecting {}'.format(
                                key, len(self.mixing_weights[key]),
                                len(self.schedule_steps)))
        self.next_threshold = 0

    def _list(self, val, repeat=None):
        if isinstance(val, int):
            if repeat:
                return [val] * repeat
            else:
                return [val]
        return list(val)

    def __call__(self, i):
        if self.next_threshold is None:
            # no adjustments left
            return None
        if i < self.next_threshold:
            # no adjustment yet
            return None
        new_weights = None
        while self.next_threshold is not None and self.next_threshold <= i:
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


class TaskMixer():
    def __init__(self, data_config, task_streams, bucket_size=2048):
        self.data_config = data_config
        self.task_streams = task_streams
        self.keys = sorted(self.task_streams.keys())

        self.schedule = MixingWeightSchedule(data_config, self.keys)
        self.current_weights = None
        min_bucket_size = self.schedule.min_bucket_size()
        if bucket_size < min_bucket_size:
            bucket_size = min_bucket_size
            logger.info('increased bucket_size to {}'.format(bucket_size))
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
            logger.info('*** mb %s set weights to %s',
                        i, list(zip(self.keys, self.current_weights)))
        self.mixed = weighted_roundrobin(
            [self.task_streams[key] for key in self.keys],
            self.current_weights)


def build_mixer(data_config, transforms, is_train=True, bucket_size=1):
    split = 'train' if is_train else 'valid'
    task_epochs = {}
    task_streams = {}
    for task in data_config['tasks']:
        if data_config['tasks'][task]['split'] != split:
            continue
        task_epoch = TaskEpoch(data_config, task)
        if is_train:
            stream = yield_infinite(task_epoch,
                                    task,
                                    transforms[task],
                                    is_train=is_train)
        else:
            # yield validation data only once
            stream = yield_once(task_epoch,
                                task,
                                transforms[task],
                                is_train=is_train)
        task_epochs[task] = task_epoch
        task_streams[task] = stream
    mixer = TaskMixer(data_config, task_streams, bucket_size=bucket_size)
    return mixer, task_epochs
