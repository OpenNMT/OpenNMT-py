#!/usr/bin/env python
"""Get vocabulary coutings from transformed corpora samples."""
import os
from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import set_random_seed, check_path
from onmt.utils.parse import ArgumentParser
from onmt.opts import dynamic_prepare_opts
from onmt.inputters.text_corpus import build_corpora_iters, get_corpora
from onmt.inputters.text_utils import process
from onmt.transforms import make_transforms, get_transforms_cls
from onmt.constants import CorpusName, CorpusTask
from collections import Counter, defaultdict
import multiprocessing as mp


def write_files_from_queues(sample_path, queues):
    """
    Standalone process that reads data from
    queues in order and write to sample files.
    """
    os.makedirs(sample_path, exist_ok=True)
    for c_name in queues.keys():
        dest_base = os.path.join(
            sample_path, "{}.{}".format(c_name, CorpusName.SAMPLE))
        with open(dest_base + ".src", 'w', encoding="utf-8") as f_src,\
                open(dest_base + ".tgt", 'w', encoding="utf-8") as f_tgt:
            while True:
                _next = False
                for q in queues[c_name]:
                    item = q.get()
                    if item == "blank":
                        continue
                    if item == "break":
                        _next = True
                        break
                    _, src_line, tgt_line = item
                    f_src.write(src_line + '\n')
                    f_tgt.write(tgt_line + '\n')
                if _next:
                    break


# Just for debugging purposes
# It appends features to subwords when dumping to file
def append_features_to_example(example, features):
    ex_toks = example.split(' ')
    feat_toks = features.split(' ')
    toks = [f"{subword}￨{feat}" for subword, feat in
            zip(ex_toks, feat_toks)]
    return " ".join(toks)


def build_sub_vocab(corpora, transforms, opts, n_sample, stride, offset):
    """Build vocab on (strided) subpart of the data."""
    sub_counter_src = Counter()
    sub_counter_tgt = Counter()
    sub_counter_src_feats = defaultdict(Counter)
    datasets_iterables = build_corpora_iters(
        corpora, transforms, opts.data,
        skip_empty_level=opts.skip_empty_level,
        stride=stride, offset=offset)
    for c_name, c_iter in datasets_iterables.items():
        for i, item in enumerate(c_iter):
            maybe_example = process(CorpusTask.TRAIN, item)
            if maybe_example is None:
                if opts.dump_samples:
                    build_sub_vocab.queues[c_name][offset].put("blank")
                continue
            src_line, tgt_line = (maybe_example['src']['src'],
                                  maybe_example['tgt']['tgt'])
            src_line_pretty = src_line
            for feat_name, feat_line in maybe_example["src"].items():
                if feat_name not in ["src", "src_original"]:
                    sub_counter_src_feats[feat_name].update(
                        feat_line.split(' '))
                    if opts.dump_samples:
                        src_line_pretty = append_features_to_example(
                            src_line_pretty, feat_line)
            sub_counter_src.update(src_line.split(' '))
            sub_counter_tgt.update(tgt_line.split(' '))
            if opts.dump_samples:
                build_sub_vocab.queues[c_name][offset].put(
                    (i, src_line_pretty, tgt_line))
            if n_sample > 0 and ((i+1) * stride + offset) >= n_sample:
                if opts.dump_samples:
                    build_sub_vocab.queues[c_name][offset].put("break")
                break
        if opts.dump_samples:
            build_sub_vocab.queues[c_name][offset].put("break")
    return sub_counter_src, sub_counter_tgt, sub_counter_src_feats


def init_pool(queues):
    """Add the queues as attribute of the pooled function."""
    build_sub_vocab.queues = queues


def build_vocab(opts, transforms, n_sample=3):
    """Build vocabulary from data."""

    if n_sample == -1:
        logger.info(f"n_sample={n_sample}: Build vocab on full datasets.")
    elif n_sample > 0:
        logger.info(f"Build vocab on {n_sample} transformed examples/corpus.")
    else:
        raise ValueError(f"n_sample should > 0 or == -1, get {n_sample}.")

    if opts.dump_samples:
        logger.info("The samples on which the vocab is built will be "
                    "dumped to disk. It may slow down the process.")
    corpora = get_corpora(opts, task=CorpusTask.TRAIN)
    counter_src = Counter()
    counter_tgt = Counter()
    counter_src_feats = defaultdict(Counter)
    from functools import partial
    queues = {c_name: [mp.Queue(opts.vocab_sample_queue_size)
                       for i in range(opts.num_threads)]
              for c_name in corpora.keys()}
    sample_path = os.path.join(
        os.path.dirname(opts.save_data), CorpusName.SAMPLE)
    if opts.dump_samples:
        write_process = mp.Process(
            target=write_files_from_queues,
            args=(sample_path, queues),
            daemon=True)
        write_process.start()
    with mp.Pool(opts.num_threads, init_pool, [queues]) as p:
        func = partial(
            build_sub_vocab, corpora, transforms,
            opts, n_sample, opts.num_threads)
        for sub_counter_src, sub_counter_tgt, sub_counter_src_feats in p.imap(
                func, range(0, opts.num_threads)):
            counter_src.update(sub_counter_src)
            counter_tgt.update(sub_counter_tgt)
            counter_src_feats.update(sub_counter_src_feats)
    if opts.dump_samples:
        write_process.join()
    return counter_src, counter_tgt, counter_src_feats


def build_vocab_main(opts):
    """Apply transforms to samples of specified data and build vocab from it.

    Transforms that need vocab will be disabled in this.
    Built vocab is saved in plain text format as following and can be pass as
    `-src_vocab` (and `-tgt_vocab`) when training:
    ```
    <tok_0>\t<count_0>
    <tok_1>\t<count_1>
    ```
    """

    ArgumentParser.validate_prepare_opts(opts, build_vocab_only=True)
    assert opts.n_sample == -1 or opts.n_sample > 1, \
        f"Illegal argument n_sample={opts.n_sample}."

    logger = init_logger()
    set_random_seed(opts.seed, False)
    transforms_cls = get_transforms_cls(opts._all_transform)

    transforms = make_transforms(opts, transforms_cls, None)

    logger.info(f"Counter vocab from {opts.n_sample} samples.")
    src_counter, tgt_counter, src_feats_counter = build_vocab(
        opts, transforms, n_sample=opts.n_sample)

    logger.info(f"Counters src:{len(src_counter)}")
    logger.info(f"Counters tgt:{len(tgt_counter)}")
    for feat_name, feat_counter in src_feats_counter.items():
        logger.info(f"Counters {feat_name}:{len(feat_counter)}")

    def save_counter(counter, save_path):
        check_path(save_path, exist_ok=opts.overwrite, log=logger.warning)
        with open(save_path, "w", encoding="utf8") as fo:
            for tok, count in counter.most_common():
                fo.write(tok + "\t" + str(count) + "\n")

    if opts.share_vocab:
        src_counter += tgt_counter
        tgt_counter = src_counter
        logger.info(f"Counters after share:{len(src_counter)}")
        save_counter(src_counter, opts.src_vocab)
    else:
        save_counter(src_counter, opts.src_vocab)
        save_counter(tgt_counter, opts.tgt_vocab)

    for k, v in src_feats_counter.items():
        save_counter(v, opts.src_feats_vocab[k])


def _get_parser():
    parser = ArgumentParser(description='build_vocab.py')
    dynamic_prepare_opts(parser, build_vocab_only=True)
    return parser


def main():
    parser = _get_parser()
    opts, unknown = parser.parse_known_args()
    build_vocab_main(opts)


if __name__ == '__main__':
    main()
