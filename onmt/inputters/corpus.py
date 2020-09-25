"""Module that contain shard utils for dynamic data."""
import os
from onmt.utils.logging import logger
from onmt.constants import CorpusName
from onmt.transforms import TransformPipe

from collections import Counter
from contextlib import contextmanager


@contextmanager
def exfile_open(filename, *args, **kwargs):
    """Extended file opener enables open(filename=None).

    This context manager enables open(filename=None) as well as regular file.
    filename None will produce endlessly None for each iterate,
    while filename with valid path will produce lines as usual.

    Args:
        filename (str|None): a valid file path or None;
        *args: args relate to open file using codecs;
        **kwargs: kwargs relate to open file using codecs.

    Yields:
        `None` repeatly if filename==None,
        else yield from file specified in `filename`.
    """
    if filename is None:
        from itertools import repeat
        _file = repeat(None)
    else:
        import codecs
        _file = codecs.open(filename, *args, **kwargs)
    yield _file
    if filename is not None and _file:
        _file.close()


class ParallelCorpus(object):
    """A parallel corpus file pair that can be loaded to iterate."""

    def __init__(self, name, src, tgt, align=None):
        """Initialize src & tgt side file path."""
        self.id = name
        self.src = src
        self.tgt = tgt
        self.align = align

    def load(self, offset=0, stride=1):
        """
        Load file and iterate by lines.
        `offset` and `stride` allow to iterate only on every
        `stride` example, starting from `offset`.
        """
        with exfile_open(self.src, mode='rb') as fs,\
                exfile_open(self.tgt, mode='rb') as ft,\
                exfile_open(self.align, mode='rb') as fa:
            logger.info(f"Loading {repr(self)}...")
            for i, (sline, tline, align) in enumerate(zip(fs, ft, fa)):
                if (i % stride) == offset:
                    sline = sline.decode('utf-8')
                    tline = tline.decode('utf-8')
                    example = {
                        'src': sline,
                        'tgt': tline
                    }
                    if align is not None:
                        example['align'] = align.decode('utf-8')
                    yield example

    def __repr__(self):
        cls_name = type(self).__name__
        return '{}({}, {}, align={})'.format(
            cls_name, self.src, self.tgt, self.align)


def get_corpora(opts, is_train=False):
    corpora_dict = {}
    if is_train:
        for corpus_id, corpus_dict in opts.data.items():
            if corpus_id != CorpusName.VALID:
                corpora_dict[corpus_id] = ParallelCorpus(
                    corpus_id,
                    corpus_dict["path_src"],
                    corpus_dict["path_tgt"],
                    corpus_dict["path_align"])
    else:
        if CorpusName.VALID in opts.data.keys():
            corpora_dict[CorpusName.VALID] = ParallelCorpus(
                CorpusName.VALID,
                opts.data[CorpusName.VALID]["path_src"],
                opts.data[CorpusName.VALID]["path_tgt"],
                opts.data[CorpusName.VALID]["path_align"])
        else:
            return None
    return corpora_dict


class ParallelCorpusIterator(object):
    """An iterator dedicate for ParallelCorpus.

    Args:
        corpus (ParallelCorpus): corpus to iterate;
        transform (Transform): transforms to be applied to corpus;
        infinitely (bool): True to iterate endlessly;
        skip_empty_level (str): security level when encouter empty line;
        stride (int): iterate corpus with this line stride;
        offset (int): iterate corpus with this line offset.
    """

    def __init__(self, corpus, transform, infinitely=False,
                 skip_empty_level='warning', stride=1, offset=0):
        self.cid = corpus.id
        self.corpus = corpus
        self.transform = transform
        self.infinitely = infinitely
        if skip_empty_level not in ['silent', 'warning', 'error']:
            raise ValueError(
                f"Invalid argument skip_empty_level={skip_empty_level}")
        self.skip_empty_level = skip_empty_level
        self.stride = stride
        self.offset = offset

    def _tokenize(self, stream):
        for example in stream:
            src = example['src'].strip('\n').split()
            tgt = example['tgt'].strip('\n').split()
            example['src'], example['tgt'] = src, tgt
            if 'align' in example:
                example['align'] = example['align'].strip('\n').split()
            yield example

    def _transform(self, stream):
        for example in stream:
            # NOTE: moved to DatasetAdapter._process method in iterator.py
            # item = self.transform.apply(
            # example, is_train=self.infinitely, corpus_name=self.cid)
            item = (example, self.transform, self.cid)
            if item is not None:
                yield item
        report_msg = self.transform.stats()
        if report_msg != '':
            logger.info("Transform statistics for {}:\n{}".format(
                self.cid, report_msg))

    def _add_index(self, stream):
        for i, item in enumerate(stream):
            example = item[0]
            line_number = i * self.stride + self.offset
            example['indices'] = line_number
            if (len(example['src']) == 0 or len(example['tgt']) == 0 or
                    ('align' in example and example['align'] == 0)):
                # empty example: skip
                empty_msg = f"Empty line exists in {self.cid}#{line_number}."
                if self.skip_empty_level == 'error':
                    raise IOError(empty_msg)
                elif self.skip_empty_level == 'warning':
                    logger.warning(empty_msg)
                continue
            yield item

    def _iter_corpus(self):
        corpus_stream = self.corpus.load(
            stride=self.stride, offset=self.offset)
        tokenized_corpus = self._tokenize(corpus_stream)
        transformed_corpus = self._transform(tokenized_corpus)
        indexed_corpus = self._add_index(transformed_corpus)
        yield from indexed_corpus

    def __iter__(self):
        if self.infinitely:
            while True:
                _iter = self._iter_corpus()
                yield from _iter
        else:
            yield from self._iter_corpus()


def build_corpora_iters(corpora, transforms, corpora_info, is_train=False,
                        skip_empty_level='warning', stride=1, offset=0):
    """Return `ParallelCorpusIterator` for all corpora defined in opts."""
    corpora_iters = dict()
    for c_id, corpus in corpora.items():
        c_transform_names = corpora_info[c_id].get('transforms', [])
        corpus_transform = [transforms[name] for name in c_transform_names]
        transform_pipe = TransformPipe.build_from(corpus_transform)
        logger.info(f"{c_id}'s transforms: {str(transform_pipe)}")
        corpus_iter = ParallelCorpusIterator(
            corpus, transform_pipe, infinitely=is_train,
            skip_empty_level=skip_empty_level, stride=stride, offset=offset)
        corpora_iters[c_id] = corpus_iter
    return corpora_iters


def save_transformed_sample(opts, transforms, n_sample=3, build_vocab=False):
    """Save transformed data sample as specified in opts."""

    if n_sample == -1:
        logger.info(f"n_sample={n_sample}: Save full transformed corpus.")
    elif n_sample == 0:
        logger.info(f"n_sample={n_sample}: no sample will be saved.")
        return
    elif n_sample > 0:
        logger.info(f"Save {n_sample} transformed example/corpus.")
    else:
        raise ValueError(f"n_sample should >= -1, get {n_sample}.")

    from onmt.inputters.dynamic_iterator import DatasetAdapter
    corpora = get_corpora(opts, is_train=True)
    if build_vocab:
        counter_src = Counter()
        counter_tgt = Counter()
    datasets_iterables = build_corpora_iters(
        corpora, transforms, opts.data, is_train=False,
        skip_empty_level=opts.skip_empty_level)
    sample_path = os.path.join(
        os.path.dirname(opts.save_data), CorpusName.SAMPLE)
    os.makedirs(sample_path, exist_ok=True)
    for c_name, c_iter in datasets_iterables.items():
        dest_base = os.path.join(
            sample_path, "{}.{}".format(c_name, CorpusName.SAMPLE))
        with open(dest_base + ".src", 'w', encoding="utf-8") as f_src,\
                open(dest_base + ".tgt", 'w', encoding="utf-8") as f_tgt:
            for i, item in enumerate(c_iter):
                maybe_example = DatasetAdapter._process(item, is_train=True)
                if maybe_example is None:
                    continue
                src_line, tgt_line = maybe_example['src'], maybe_example['tgt']
                if build_vocab:
                    counter_src.update(src_line.split(' '))
                    counter_tgt.update(tgt_line.split(' '))
                f_src.write(src_line + '\n')
                f_tgt.write(tgt_line + '\n')
                if n_sample > 0 and i >= n_sample:
                    break
    if build_vocab:
        return counter_src, counter_tgt
