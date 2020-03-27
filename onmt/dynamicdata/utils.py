import itertools
import subprocess

UNDER = '▁'


def safe_zip(*iterables):
    iters = [iter(x) for x in iterables]
    sentinel = object()
    for (j, tpl) in enumerate(itertools.zip_longest(*iterables,
                                                    fillvalue=sentinel)):
        for (i, val) in enumerate(tpl):
            if val is sentinel:
                raise ValueError(
                    'Column {} was too short. '
                    'Row {} (and later) missing.'.format(i, j))
        yield tpl


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))


def weighted_roundrobin(streams, weights):
    repeated = []
    for stream, weight in safe_zip(streams, weights):
        for _ in range(weight):
            repeated.append(stream)
    yield from roundrobin(*repeated)


def external_linecount(file_path):
    if file_path.endswith('.gz'):
        ext_lc = subprocess.check_output(
            ['zcat {} | wc -l'.format(file_path)], shell=True).split()[0]
    else:
        ext_lc = subprocess.check_output(['wc', '-l', file_path]).split()[0]
    ext_lc = int(ext_lc.decode('utf-8'))
    return ext_lc
