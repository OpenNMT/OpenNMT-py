# -*- coding: utf-8 -*-

import codecs
import os

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

    @staticmethod
    def sort_key(ex):
        """ Sort using the size of the image: (width, height)."""
        return ex.src.size(2), ex.src.size(1)

    @classmethod
    def _make_example(cls, pair, truncate=None, **kwargs):
        """
        Args:
            path (str): location of a src file containing image paths
            src_dir (str): location of source images
            side (str): 'src' or 'tgt'
            truncate: maximum img size ((0,0) or None for unlimited)

        Yields:
            a dictionary containing image data, path and index for each line.
        """
        img, filename = pair
        # problem is that the value of truncate getting passed is an int,
        # which is not subscriptable
        # gonna ignore the image truncating logic for now...
        # truncating doesn't really appear to truncate for images, it
        # filters out images that are too big.
        """
        if truncate and truncate != (0, 0):
            if not (img.size(1) <= truncate[0]
                    and img.size(2) <= truncate[1]):
                continue
        """
        return {'src': img, 'src_path': filename}

    @classmethod
    def _make_iterator_from_file(cls, path, directory, **kwargs):
        """
        Args:
            path(str):
            src_dir(str):

        Yields:
            img: and image tensor
            filename(str): the image filename
        """
        assert directory is not None and os.path.exists(directory), \
            'src_dir must be a valid directory if data_type is {}'.format(
            cls.data_type)

        from PIL import Image
        from torchvision import transforms

        with codecs.open(path, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                filename = line.strip()
                img_path = os.path.join(directory, filename)
                if not os.path.exists(img_path):
                    img_path = line

                assert os.path.exists(img_path), \
                    'img path %s not found' % (line.strip())

                img = transforms.ToTensor()(Image.open(img_path))
                yield img, filename
