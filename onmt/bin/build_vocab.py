#!/usr/bin/env python
"""Get vocabulary coutings from transformed corpora samples."""
from onmt.utils.logging import init_logger
from onmt.utils.misc import set_random_seed, check_path
from onmt.utils.parse import ArgumentParser
from onmt.opts import dynamic_prepare_opts
from onmt.inputters.corpus import build_vocab
from onmt.transforms import make_transforms, get_transforms_cls


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
    fields = None

    transforms = make_transforms(opts, transforms_cls, fields)

    logger.info(f"Counter vocab from {opts.n_sample} samples.")
    src_counter, tgt_counter = build_vocab(
        opts, transforms, n_sample=opts.n_sample)

    logger.info(f"Counters src:{len(src_counter)}")
    logger.info(f"Counters tgt:{len(tgt_counter)}")

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
