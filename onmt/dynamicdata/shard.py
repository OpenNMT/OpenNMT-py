import collections
import gzip
import itertools
import math
import os
import random

from .vocab import SimpleSharedVocabulary
from .utils import *

from onmt.utils.logging import logger

def open_for_reading(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'tr')
    else:
        return open(path, 'r')


def para_reader(input):
    with open_for_reading(input['src']) as src_fobj:
        with open_for_reading(input['tgt']) as tgt_fobj:
            while True:
                src = next(src_fobj, None)
                tgt = next(tgt_fobj, None)
                if src is None and tgt is None:
                    break
                elif src is None:
                    raise Exception('src {} ended before tgt {}'.format(
                        input['src'], input['tgt']))
                elif tgt is None:
                    raise Exception('tgt {} ended before src {}'.format(
                        input['tgt'], input['src']))
                yield src, tgt


def mono_reader(input):
    with open_for_reading(input['mono']) as fobj:
        for line in fobj:
            yield (line,)


def check_exist(input):
    if 'mono' in input:
        if not os.path.exists(input['mono']):
            raise Exception('Missing: {}'.format(input['mono']))
    else:
        for side in ('src', 'tgt'):
            if not os.path.exists(input[side]):
                raise Exception('Missing: {}'.format(input[side]))


def pretokenize(stream, tokenizer):
    for tpl in stream:
        out = []
        for side in tpl:
            tokens, feats = tokenizer.tokenize(side)
            out.append(' '.join(tokens) + '\n')
        yield tuple(out)


def predetokenize(stream):
    for tpl in stream:
        out = []
        for side in tpl:
            side = side.replace(' ', '')
            side = side.lstrip(UNDER)
            side = side.replace(UNDER, ' ')
            out.append(side)
        yield tuple(out)


def adjust_shard_size(total, max_shard_size, max_initial_shards):
    one_bucket = max_shard_size * max_initial_shards
    initial_shards = max_initial_shards
    shard_size = max_shard_size
    if total < one_bucket:
        # small data: reduce number of shards
        initial_shards = int(math.ceil(total / max_shard_size))
    else:
        # large data: balance shard size
        buckets = int(math.ceil(total / one_bucket))
        shard_size = int(math.ceil(total / (buckets * initial_shards)))
    initial_shards = min(initial_shards, max_initial_shards)
    shard_size = min(shard_size, max_shard_size)
    return initial_shards, shard_size


class DataSharder():
    def __init__(self, data_config,
                 max_shard_size, max_initial_shards,
                 compress=True, vocab_counter=None,
                 pre=None):
        self.data_config = data_config
        self.max_shard_size = max_shard_size
        self.max_initial_shards = max_initial_shards
        self.compress = compress
        self.vocab_counter = vocab_counter
        if pre == 'tokenize':
            import pyonmttok
            tokenizer = pyonmttok.Tokenizer(
                'aggressive',
                joiner=UNDER,
                joiner_annotate=False,
                spacer_annotate=True,
                segment_alphabet_change=True)
            self.tokenize = lambda stream: pretokenize(stream, tokenizer)
        elif pre == 'detokenize':
            self.tokenize = predetokenize
        else:
            self.tokenize = None

        self._open_shards = []
        self._last_shard = None

    def __call__(self):
        for task in self.data_config['tasks']:
            for input in self.data_config['tasks'][task]['_inputs']:
                check_exist(self.data_config['inputs'][input])
        os.makedirs(
            self.data_config['meta']['shard']['rootdir'],
            exist_ok=True)
        for task in self.data_config['tasks']:
            self.shard_task(task)

    def shard_task(self, task):
        taskdir = os.path.join(self.data_config['meta']['shard']['rootdir'],
                               'shards', task)
        os.makedirs(taskdir, exist_ok=True)
        task_type = self.data_config['tasks'][task]['type']
        if task_type == 'para':
            shard_cls = ParaShard
            reader_func = para_reader
        elif task_type == 'mono':
            shard_cls = MonoShard
            reader_func = mono_reader
        else:
            raise Exception('Unrecognized task type "{}"'.format(task_type))
        # make balanced shards
        total = self.data_config['tasks'][task]['_size']
        initial_shards, shard_size = adjust_shard_size(
            total, self.max_shard_size, self.max_initial_shards)
        logger.info('Task {task}: total {total},'
                    ' initial_shards {initial_shards},'
                    ' shard_size {shard_size},'
                    ' product {prod}'.format(task=task,
                                             total=total,
                                             initial_shards=initial_shards,
                                             shard_size=shard_size,
                                             prod=initial_shards*shard_size))
        # create shards that are transparently reopened when filled
        n_shards = self.data_config['tasks'][task].get('n_shards',
                                                       initial_shards)
        self._open_shards = [
            shard_cls(self, task, taskdir, shard_size, self.compress)
            for _ in range(n_shards)]
        self._last_shard = -1

        # roundrobin the corpora
        # readers are repeated based on size to balance shards
        streams = []
        weights = []
        for input in self.data_config['tasks'][task]['_inputs']:
            streams.append(reader_func(self.data_config['inputs'][input]))
            weights.append(self.data_config['inputs'][input]['size'])
        stream = weighted_roundrobin(streams, weights)
        if self.tokenize is not None:
            stream = self.tokenize(stream)
        while True:
            # read a bucket
            bucket = list(itertools.islice(stream, initial_shards * 100))
            if len(bucket) == 0:
                break
            # permute
            random.shuffle(bucket)
            # write
            for tpl, shard in zip(bucket, itertools.cycle(self._open_shards)):
                if self.vocab_counter:
                    # word count, if desired
                    self.vocab_counter.add(task, tpl)
                shard.write(tpl)
        for shard in self._open_shards:
            shard.close()


class Shard():
    def __init__(self, sharder, task, taskdir, max_shard_size, compress=False):
        self.sharder = sharder
        self.task = task
        self.taskdir = taskdir
        self.max_shard_size = max_shard_size
        self.compress = compress
        self.fobjs = None
        self.index = None
        self.count = 0

    def _reset(self):
        self.close()
        self.sharder._last_shard += 1
        self.index = self.sharder._last_shard
        self.count = 0
        self._open()

    def _open_single(self, suffix):
        # TODO: check for valid suffix
        ext = '.gz' if self.compress else ''
        path = os.path.join(
            self.taskdir,
            '{index}.{suffix}{ext}'.format(
                index=self.index,
                suffix=suffix,
                ext=ext))
        if self.compress:
            return gzip.open(path, 'tw')
        else:
            return open(path, 'w')

    def write(self, tpl):
        if self.fobjs is None or self.count == self.max_shard_size:
            self._reset()
        self._write_helper(tpl)
        self.count += 1

    def close(self):
        if self.fobjs is not None:
            for fobj in self.fobjs:
                fobj.close()


class MonoShard(Shard):
    def _open(self):
        self.fobjs = [self._open_single('mono')]

    def _write_helper(self, tpl):
        self.fobjs[0].write(tpl[0])


class ParaShard(Shard):
    def _open(self):
        self.fobjs = [self._open_single(side)
                      for side in ('src', 'tgt')]

    def _write_helper(self, tpl):
        for line, fobj in zip(tpl, self.fobjs):
            fobj.write(line)
