"""Module that contain shard utils for dynamic data."""
import os
from onmt.utils.logging import logger
from onmt.constants import CorpusName, CorpusTask
from onmt.transforms import TransformPipe
from onmt.inputters.text_utils import process, parse_features, append_features_to_text
from contextlib import contextmanager
import itertools


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


class BlockwiseCorpus(object):
    """A corpus class for reading a single file block by block."""

    def __init__(self, name, file_path, block_size=4096):
        """Initialize file path and block size."""
        self.id = name
        self.file_path = file_path
        self.block_size = block_size

    def load(self, offset=0, stride=1):
        """
        Load file and iterate by blocks.
        `offset` and `stride` allow iterating only on every
        `stride` block, starting from `offset`.
        """

        def make_ex(block_content):
            example = {
                "src": block_content,
                "tgt": block_content,
                "src_original": block_content,
                "tgt_original": block_content,
            }
            return example

        with open(self.file_path, mode="r", encoding="utf-8") as file:
            block_content = ""
            block_index = 0

            while True:
                chunk = file.read(self.block_size)
                if not chunk:
                    break

                if (block_index // stride) % stride == offset:
                    block_content += chunk

                    if len(chunk) < self.block_size:
                        # Reached end of file
                        yield make_ex(block_content)
                        break

                    if len(block_content) >= self.block_size:
                        yield make_ex(block_content)
                block_content = ""
                block_index += 1

    def __str__(self):
        cls_name = type(self).__name__
        return (
            f"{cls_name}({self.id}, {self.file_path}, {self.file_path}"
            f"align={None}, "
            f"n_src_feats={0}, "
            f'src_feats_defaults="{None}")'
        )


class ParallelCorpus(object):
    """A parallel corpus file pair that can be loaded to iterate."""

    def __init__(
        self, name, src, tgt, align=None, n_src_feats=0, src_feats_defaults=None
    ):
        """Initialize src & tgt side file path."""
        self.id = name
        self.src = src
        self.tgt = tgt
        self.align = align
        self.n_src_feats = n_src_feats
        self.src_feats_defaults = src_feats_defaults

    def load(self, offset=0, stride=1):
        """
        Load file and iterate by lines.
        `offset` and `stride` allow to iterate only on every
        `stride` example, starting from `offset`.
        """

        def make_ex(sline, tline, align):
            sline, sfeats = parse_features(
                sline,
                n_feats=self.n_src_feats,
                defaults=self.src_feats_defaults,
            )
            # 'src_original' and 'tgt_original' store the
            # original line before tokenization. These
            # fields are used later on in the feature
            # transforms.
            example = {
                "src": sline,
                "tgt": tline,
                "src_original": sline,
                "tgt_original": tline,
            }
            if align is not None:
                example["align"] = align
            if sfeats is not None:
                example["src_feats"] = [f for f in sfeats]
            return example

        if isinstance(self.src, list):
            fs = self.src
            ft = [] if self.tgt is None else self.tgt
            fa = [] if self.align is None else self.align
            for i, (sline, tline, align) in enumerate(
                itertools.zip_longest(fs, ft, fa)
            ):
                if (i // stride) % stride == offset:
                    yield make_ex(sline, tline, align)
        else:
            with exfile_open(self.src, mode="rb") as fs, exfile_open(
                self.tgt, mode="rb"
            ) as ft, exfile_open(self.align, mode="rb") as fa:
                for i, (sline, tline, align) in enumerate(zip(fs, ft, fa)):
                    if (i // stride) % stride == offset:
                        if tline is not None:
                            tline = tline.decode("utf-8")
                        if align is not None:
                            align = align.decode("utf-8")
                        yield make_ex(sline.decode("utf-8"), tline, align)

    def __str__(self):
        cls_name = type(self).__name__
        return (
            f"{cls_name}({self.id}, {self.src}, {self.tgt}, "
            f"align={self.align}, "
            f"n_src_feats={self.n_src_feats}, "
            f'src_feats_defaults="{self.src_feats_defaults}")'
        )


def get_corpora(opts, task=CorpusTask.TRAIN, src=None, tgt=None, align=None):
    corpora_dict = {}
    if task == CorpusTask.TRAIN:
        for corpus_id, corpus_dict in opts.data.items():
            if corpus_id != CorpusName.VALID:
                if corpus_dict.get("path_txt", None) is None:
                    corpora_dict[corpus_id] = ParallelCorpus(
                        corpus_id,
                        corpus_dict["path_src"],
                        corpus_dict["path_tgt"],
                        corpus_dict["path_align"],
                        n_src_feats=opts.n_src_feats,
                        src_feats_defaults=opts.src_feats_defaults,
                    )
                else:
                    corpora_dict[corpus_id] = BlockwiseCorpus(
                        corpus_id,
                        corpus_dict["path_txt"],
                        block_size=8192,  # number of characters
                    )
    elif task == CorpusTask.VALID:
        if CorpusName.VALID in opts.data.keys():
            corpora_dict[CorpusName.VALID] = ParallelCorpus(
                CorpusName.VALID,
                opts.data[CorpusName.VALID]["path_src"],
                opts.data[CorpusName.VALID]["path_tgt"] if tgt is None else None,
                opts.data[CorpusName.VALID]["path_align"],
                n_src_feats=opts.n_src_feats,
                src_feats_defaults=opts.src_feats_defaults,
            )
        else:
            return None
    else:
        corpora_dict[CorpusName.INFER] = ParallelCorpus(
            CorpusName.INFER,
            src if src else opts.src,
            tgt if tgt else opts.tgt,
            align if align else None,
            n_src_feats=opts.n_src_feats,
            src_feats_defaults=opts.src_feats_defaults,
        )
    return corpora_dict


class ParallelCorpusIterator(object):
    """An iterator dedicated to ParallelCorpus.

    Args:
        corpus (ParallelCorpus): corpus to iterate;
        transform (TransformPipe): transforms to be applied to corpus;
        skip_empty_level (str): security level when encouter empty line;
        stride (int): iterate corpus with this line stride;
        offset (int): iterate corpus with this line offset.
    """

    def __init__(
        self, corpus, transform, skip_empty_level="warning", stride=1, offset=0
    ):
        self.cid = corpus.id
        self.corpus = corpus
        self.transform = transform
        if skip_empty_level not in ["silent", "warning", "error"]:
            raise ValueError(f"Invalid argument skip_empty_level={skip_empty_level}")
        self.skip_empty_level = skip_empty_level
        self.stride = stride
        self.offset = offset

    def _process(self, stream):
        for i, example in enumerate(stream):
            example["src"] = example["src"].strip().split(" ")
            example["src_original"] = example["src_original"].strip().split(" ")
            if "src_feats" in example:
                example["src_feats"] = [
                    feat.strip().split(" ") for feat in example["src_feats"]
                ]
            line_number = i * self.stride + self.offset
            example["cid_line_number"] = line_number
            example["cid"] = self.cid
            if "align" in example:
                example["align"] = example["align"].strip().split(" ")
            if example["tgt"] is not None:
                example["tgt"] = example["tgt"].strip().split(" ")
                example["tgt_original"] = example["tgt_original"].strip().split(" ")
                if (
                    len(example["src"]) == 0
                    or len(example["tgt"]) == 0
                    or ("align" in example and example["align"] == 0)
                ):
                    # empty example: skip
                    empty_msg = f"Empty line  in {self.cid}#{line_number}."
                    if self.skip_empty_level == "error":
                        raise IOError(empty_msg)
                    elif self.skip_empty_level == "warning":
                        logger.warning(empty_msg)
                    if len(example["src"]) == 0 and len(example["tgt"]) == 0:
                        yield (example, self.transform, self.cid)
                    continue
            yield (example, self.transform, self.cid)
        report_msg = self.transform.stats()
        if report_msg != "":
            logger.info(
                "* Transform statistics for {}({:.2f}%):\n{}\n".format(
                    self.cid, 100 / self.stride, report_msg
                )
            )

    def __iter__(self):
        corpus_stream = self.corpus.load(stride=self.stride, offset=self.offset)
        corpus = self._process(corpus_stream)
        yield from corpus


def build_corpora_iters(
    corpora, transforms, corpora_info, skip_empty_level="warning", stride=1, offset=0
):
    """Return `ParallelCorpusIterator` for all corpora defined in opts."""
    corpora_iters = dict()
    for c_id, corpus in corpora.items():
        transform_names = corpora_info[c_id].get("transforms", [])
        corpus_transform = [
            transforms[name] for name in transform_names if name in transforms
        ]
        transform_pipe = TransformPipe.build_from(corpus_transform)
        corpus_iter = ParallelCorpusIterator(
            corpus,
            transform_pipe,
            skip_empty_level=skip_empty_level,
            stride=stride,
            offset=offset,
        )
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
        corpora, transforms, opts.data, skip_empty_level=opts.skip_empty_level
    )
    sample_path = os.path.join(os.path.dirname(opts.save_data), CorpusName.SAMPLE)
    os.makedirs(sample_path, exist_ok=True)
    for c_name, c_iter in datasets_iterables.items():
        dest_base = os.path.join(sample_path, "{}.{}".format(c_name, CorpusName.SAMPLE))
        with open(dest_base + ".src", "w", encoding="utf-8") as f_src, open(
            dest_base + ".tgt", "w", encoding="utf-8"
        ) as f_tgt:
            bucket = []
            for i, ex in enumerate(c_iter):
                if i > n_sample:
                    break
                else:
                    bucket.append(ex)
            pro_bucket = process(CorpusTask.TRAIN, bucket)
            if pro_bucket is not None:
                for maybe_example in pro_bucket:
                    if maybe_example is not None:
                        src_line, tgt_line = (
                            maybe_example["src"]["src"],
                            maybe_example["tgt"]["tgt"],
                        )

                        if "feats" in maybe_example["src"]:
                            src_feats_lines = maybe_example["src"]["feats"]
                        else:
                            src_feats_lines = []

                        src_pretty_line = append_features_to_text(
                            src_line, src_feats_lines
                        )

                        f_src.write(src_pretty_line + "\n")
                        f_tgt.write(tgt_line + "\n")
