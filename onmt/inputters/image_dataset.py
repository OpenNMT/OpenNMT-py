# -*- coding: utf-8 -*-
"""
    ImageDataset
"""

import codecs
import os

from torchtext.data import Example

from onmt.inputters.dataset_base import DatasetBase


class ImageDataset(DatasetBase):
    """ Dataset for data_type=='img'

        Build `Example` objects, `Field` objects, and filter_pred function
        from image corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            tgt_seq_length (int): maximum target sequence length.
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """

    data_type = 'img'

    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 num_src_feats=0, num_tgt_feats=0,
                 tgt_seq_length=0, use_filter_pred=True):
        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats

        if tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        fields = [(k, fields[k]) if k in fields else (k, None) for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)
        examples = [Example.fromlist(ev, fields) for ev in example_values]

        def filter_pred(ex):
            return 0 < len(ex.tgt) <= tgt_seq_length

        filter_pred = filter_pred if use_filter_pred \
            and tgt_examples_iter is not None else None

        super(ImageDataset, self).__init__(examples, fields, filter_pred)

    @staticmethod
    def sort_key(ex):
        """ Sort using the size of the image: (width, height)."""
        return ex.src.size(2), ex.src.size(1)

    @classmethod
    def make_examples_nfeats_tpl(cls, iterator, path, directory, **kwargs):
        """
        Note: one of img_iter and img_path must be not None
        Args:
            img_iter(iterator): an iterator that yields pairs (img, filename)
                or None
            img_path(str): location of a src file containing image paths
                or None
            src_dir (str): location of source images

        Returns:
            (example_dict iterator, num_feats) tuple
        """
        # This method disregards one of its arguments. This should not be.
        if iterator is None and path is None:
            raise ValueError("'iterator' and 'path' must not both be None")

        iterator = cls.make_iterator_from_file(path, directory)
        examples_iter = cls.make_examples(iterator, directory, 'src')

        return examples_iter, 0

    @classmethod
    def make_examples(cls, img_iter, src_dir, side, truncate=None):
        """
        Args:
            path (str): location of a src file containing image paths
            src_dir (str): location of source images
            side (str): 'src' or 'tgt'
            truncate: maximum img size ((0,0) or None for unlimited)

        Yields:
            a dictionary containing image data, path and index for each line.
        """
        assert src_dir is not None and os.path.exists(src_dir), \
            'src_dir must be a valid directory if data_type is {}'.format(
            cls.data_type)

        for index, (img, filename) in enumerate(img_iter):
            if truncate and truncate != (0, 0):
                if not (img.size(1) <= truncate[0]
                        and img.size(2) <= truncate[1]):
                    continue

            # if the continue statement is reached, then the outputs of this
            # function will have missing indices
            example_dict = {side: img,
                            side + '_path': filename,
                            'indices': index}
            yield example_dict

    @classmethod
    def make_iterator_from_file(cls, path, src_dir):
        """
        Args:
            path(str):
            src_dir(str):

        Yields:
            img: and image tensor
            filename(str): the image filename
        """
        from PIL import Image
        from torchvision import transforms

        with codecs.open(path, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                filename = line.strip()
                img_path = os.path.join(src_dir, filename)
                if not os.path.exists(img_path):
                    img_path = line

                assert os.path.exists(img_path), \
                    'img path %s not found' % (line.strip())

                img = transforms.ToTensor()(Image.open(img_path))
                yield img, filename
