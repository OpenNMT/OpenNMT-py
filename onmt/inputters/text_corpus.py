"""Module that contain shard utils for dynamic data."""
import os
from onmt.utils.logging import logger
from onmt.constants import CorpusName, CorpusTask
from onmt.transforms import TransformPipe
from onmt.inputters.text_utils import process
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

    def __init__(self, name, src, tgt, align=None, src_feats=None):
        """Initialize src & tgt side file path."""
        self.id = name
        self.src = src
        self.tgt = tgt
        self.align = align
        self.src_feats = src_feats

    def load(self, offset=0, stride=1):
        """
        Load file and iterate by lines.
        `offset` and `stride` allow to iterate only on every
        `stride` example, starting from `offset`.
        """
        if self.src_feats:
            features_names = []
            features_files = []
            for feat_name, feat_path in self.src_feats.items():
                features_names.append(feat_name)
                features_files.append(open(feat_path, mode='rb'))
        else:
            features_files = []
        with exfile_open(self.src, mode='rb') as fs,\
                exfile_open(self.tgt, mode='rb') as ft,\
                exfile_open(self.align, mode='rb') as fa:
            for i, (sline, tline, align, *features) in \
                    enumerate(zip(fs, ft, fa, *features_files)):
                if (i % stride) == offset:
                    sline = sline.decode('utf-8')
                    if tline is not None:
                        tline = tline.decode('utf-8')
                    # 'src_original' and 'tgt_original' store the
                    # original line before tokenization. These
                    # fields are used later on in the feature
                    # transforms.
                    example = {
                        'src': sline,
                        'tgt': tline,
                        'src_original': sline,
                        'tgt_original': tline
                    }
                    if align is not None:
                        example['align'] = align.decode('utf-8')
                    if features:
                        example['src_feats'] = dict()
                        for j, feat in enumerate(features):
                            example['src_feats'][features_names[j]] = \
                                feat.decode("utf-8")
                    yield example
        for f in features_files:
            f.close()

    def __str__(self):
        cls_name = type(self).__name__
        return '{}({}, {}, align={}, src_feats={})'.format(
            cls_name, self.src, self.tgt, self.align, self.src_feats)


def get_corpora(opts, task=CorpusTask.TRAIN):
    corpora_dict = {}
    if task == CorpusTask.TRAIN:
        for corpus_id, corpus_dict in opts.data.items():
            if corpus_id != CorpusName.VALID:
                corpora_dict[corpus_id] = ParallelCorpus(
                    corpus_id,
                    corpus_dict["path_src"],
                    corpus_dict["path_tgt"],
                    corpus_dict["path_align"],
                    corpus_dict["src_feats"])
    elif task == CorpusTask.VALID:
        if CorpusName.VALID in opts.data.keys():
            corpora_dict[CorpusName.VALID] = ParallelCorpus(
                CorpusName.VALID,
                opts.data[CorpusName.VALID]["path_src"],
                opts.data[CorpusName.VALID]["path_tgt"],
                opts.data[CorpusName.VALID]["path_align"],
                opts.data[CorpusName.VALID]["src_feats"])
        else:
            return None
    else:
        corpora_dict[CorpusName.INFER] = ParallelCorpus(
                CorpusName.INFER,
                opts.src,
                opts.tgt,
                src_feats=opts.src_feats)
    return corpora_dict


class ParallelCorpusIterator(object):
    """An iterator dedicate for ParallelCorpus.

    Args:
        corpus (ParallelCorpus): corpus to iterate;
        transform (TransformPipe): transforms to be applied to corpus;
        skip_empty_level (str): security level when encouter empty line;
        stride (int): iterate corpus with this line stride;
        offset (int): iterate corpus with this line offset.
    """

    def __init__(self, corpus, transform,
                 skip_empty_level='warning', stride=1, offset=0):
        self.cid = corpus.id
        self.corpus = corpus
        self.transform = transform
        if skip_empty_level not in ['silent', 'warning', 'error']:
            raise ValueError(
                f"Invalid argument skip_empty_level={skip_empty_level}")
        self.skip_empty_level = skip_empty_level
        self.stride = stride
        self.offset = offset

    def _tokenize(self, stream):
        for example in stream:
            example['src'] = example['src'].strip('\n').split()
            example['src_original'] = \
                example['src_original'].strip("\n").split()
            if example['tgt'] is not None:
                example['tgt'] = example['tgt'].strip('\n').split()
                example['tgt_original'] = \
                    example['tgt_original'].strip("\n").split()
            if 'align' in example:
                example['align'] = example['align'].strip('\n').split()
            if 'src_feats' in example:
                for k in example['src_feats'].keys():
                    example['src_feats'][k] = \
                        example['src_feats'][k].strip('\n').split()
            yield example

    def _transform(self, stream):
        for example in stream:
            # NOTE: moved to dynamic_iterator.py cf process()
            # item = self.transform.apply(
            # example, is_train=self.infinitely, corpus_name=self.cid)
            item = (example, self.transform, self.cid)
            if item is not None:
                yield item
        report_msg = self.transform.stats()
        if report_msg != '':
            logger.info(
                "* Transform statistics for {}({:.2f}%):\n{}\n".format(
                    self.cid, 100/self.stride, report_msg
                )
            )

    def _add_index(self, stream):
        for i, item in enumerate(stream):
            example = item[0]
            line_number = i * self.stride + self.offset
            example['indices'] = line_number
            if example['tgt'] is not None:
                if (len(example['src']) == 0 or len(example['tgt']) == 0 or
                        ('align' in example and example['align'] == 0)):
                    # empty example: skip
                    empty_msg = f"Empty line  in {self.cid}#{line_number}."
                    if self.skip_empty_level == 'error':
                        raise IOError(empty_msg)
                    elif self.skip_empty_level == 'warning':
                        logger.warning(empty_msg)
                    continue
            yield item

    def __iter__(self):
        corpus_stream = self.corpus.load(
            stride=self.stride, offset=self.offset
        )
        tokenized_corpus = self._tokenize(corpus_stream)
        transformed_corpus = self._transform(tokenized_corpus)
        indexed_corpus = self._add_index(transformed_corpus)
        yield from indexed_corpus


def build_corpora_iters(corpora, transforms, corpora_info,
                        skip_empty_level='warning', stride=1, offset=0):
    """Return `ParallelCorpusIterator` for all corpora defined in opts."""
    corpora_iters = dict()
    for c_id, corpus in corpora.items():
        transform_names = corpora_info[c_id].get('transforms', [])
        corpus_transform = [
            transforms[name] for name in transform_names if name in transforms
        ]
        transform_pipe = TransformPipe.build_from(corpus_transform)
        corpus_iter = ParallelCorpusIterator(
            corpus, transform_pipe,
            skip_empty_level=skip_empty_level, stride=stride, offset=offset)
        corpora_iters[c_id] = corpus_iter
    return corpora_iters


def save_transformed_sample(opts, transforms, n_sample=3):
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

    corpora = get_corpora(opts, CorpusTask.TRAIN)
    datasets_iterables = build_corpora_iters(
        corpora, transforms, opts.data,
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
                maybe_example = process(CorpusTask.TRAIN, item)
                if maybe_example is None:
                    continue
                src_line, tgt_line = (maybe_example['src']['src'],
                                      maybe_example['tgt']['tgt'])
                f_src.write(src_line + '\n')
                f_tgt.write(tgt_line + '\n')
                if n_sample > 0 and i >= n_sample:
                    break
