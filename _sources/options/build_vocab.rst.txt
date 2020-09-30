Build Vocab
===========

.. argparse::
    :filename: ../onmt/bin/build_vocab.py
    :func: _get_parser
    :prog: build_vocab.py

    Transform/BART : @before
        This transform will not take effet when building vocabulary.

    Transform/SwitchOut : @before
        This transform will not take effet when building vocabulary.

    Transform/Subword/Common : @before
        Common options for transforms related to tokenization/subword.
        See :class:`~onmt.transforms.tokenize.TokenizerTransform` for more detail.
