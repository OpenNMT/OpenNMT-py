import collections
import gzip
import itertools
import os
import random

from .vocab import SimpleSharedVocabulary
from .utils import *

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

class DataSharder():
    def __init__(self, data_config,
                 max_shard_size, initial_shards,
                 compress=True, vocab_counter=None,
                 pretokenize=False):
        self.data_config = data_config
        self.max_shard_size = max_shard_size
        self.initial_shards = initial_shards
        self.compress = compress
        self.vocab_counter = vocab_counter
        if pretokenize:
            import pyonmttok
            self.tokenizer = pyonmttok.Tokenizer(
                'aggressive',
                joiner=UNDER,
                joiner_annotate=False,
                spacer_annotate=True,
                segment_alphabet_change=True)
        else:
            self.tokenizer = None

        self._open_shards = []
        self._last_shard = None

    def __call__(self):
        for group in self.data_config['groups']:
            for input in self.data_config['groups'][group]['_inputs']:
                check_exist(self.data_config['inputs'][input])
        os.makedirs(
            self.data_config['meta']['shard']['rootdir'],
            exist_ok=True)
        for group in self.data_config['groups']:
            self.shard_group(group)

    def shard_group(self, group):
        groupdir = os.path.join(self.data_config['meta']['shard']['rootdir'],
                                group)
        os.makedirs(groupdir, exist_ok=True)
        group_type = self.data_config['groups'][group]['type']
        if group_type == 'para':
            shard_cls = ParaShard
            reader_func = para_reader
        elif group_type == 'mono':
            shard_cls = MonoShard
            reader_func = mono_reader
        else:
            raise Exception('Unrecognized group type "{}"'.format(group_type))
        # create shards that are transparently reopened when filled
        n_shards = self.data_config['groups'][group].get('n_shards', self.initial_shards)
        self._open_shards = [
            shard_cls(self, group, groupdir, self.compress)
            for _ in range(n_shards)]
        self._last_shard = -1

        # roundrobin the corpora
        # readers are repeated based on size to balance shards
        streams = []
        weights = []
        for input in self.data_config['groups'][group]['_inputs']:
            streams.append(reader_func(self.data_config['inputs'][input]))
            weights.append(self.data_config['inputs'][input]['size'])
        stream = weighted_roundrobin(streams, weights)
        if self.tokenizer is not None:
            stream = pretokenize(stream, self.tokenizer)
        while True:
            # read a bucket
            bucket = list(itertools.islice(stream, self.initial_shards * 100))
            if len(bucket) == 0:
                break
            # permute
            random.shuffle(bucket)
            # write
            for tpl, shard in zip(bucket, itertools.cycle(self._open_shards)):
                if self.vocab_counter:
                    # word count, if desired
                    self.vocab_counter.add(group, tpl)
                shard.write(tpl)
        for shard in self._open_shards:
            shard.close()

class Shard():
    def __init__(self, sharder, group, groupdir, compress=False):
        self.sharder = sharder
        self.group = group
        self.groupdir = groupdir
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
            self.groupdir,
            '{index}.{suffix}{ext}'.format(
                index=self.index,
                suffix=suffix,
                ext=ext))
        if self.compress:
            return gzip.open(path, 'tw')
        else:
            return open(path, 'w')

    def write(self, tpl):
        if self.fobjs is None or self.count == self.sharder.max_shard_size:
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
