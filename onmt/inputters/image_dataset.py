# -*- coding: utf-8 -*-

import codecs
import os

from onmt.inputters.dataset_base import NonTextDatasetBase


class ImageDataset(NonTextDatasetBase):
    data_type = 'img'  # get rid of this class attribute asap

    @staticmethod
    def sort_key(ex):
        """ Sort using the size of the image: (width, height)."""
        return ex.src.size(2), ex.src.size(1)

    @staticmethod
    def make_image_examples(img_iter, img_path, img_dir, image_channel_size=3):
        """
        Note: one of img_iter and img_path must be not None
        Args:
            img_iter(iterator): an iterator that yields pairs (img, filename)
                (or None)
            img_path(str): location of a src file containing image paths
                (or None)
            src_dir (str): location of source images

        Returns:
            example_dict iterator
        """
        if img_iter is None and img_path is None:
            raise ValueError("Either img_iter or img_path must be non-None")
        if img_iter is None:
            img_iter = ImageDataset.make_iterator_from_file(
                img_path, img_dir, image_channel_size
            )

        examples_iter = ImageDataset.make_examples(img_iter, img_dir, 'src')

        return examples_iter

    @staticmethod
    def make_examples(img_iter, src_dir, side, truncate=None):
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
            'src_dir must be a valid directory if data_type is img'

        for i, (img, filename) in enumerate(img_iter):
            if truncate and truncate != (0, 0):
                if not (img.size(1) <= truncate[0]
                        and img.size(2) <= truncate[1]):
                    continue

            yield {side: img, side + '_path': filename, 'indices': i}

    @staticmethod
    def make_iterator_from_file(path, src_dir, image_channel_size=3):
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
        import cv2

        with codecs.open(path, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                filename = line.strip()
                img_path = os.path.join(src_dir, filename)
                if not os.path.exists(img_path):
                    img_path = line

                assert os.path.exists(img_path), \
                    'img path %s not found' % (line.strip())

                if image_channel_size == 1:
                    img = transforms.ToTensor()(
                        Image.fromarray(cv2.imread(img_path, 0)))
                else:
                    img = transforms.ToTensor()(Image.open(img_path))

                yield img, filename
