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
    def make_examples(path, src_dir, side, truncate=None, channel_size=3):
        """
        Args:
            path (str): location of a src file containing image paths
            src_dir (str): location of source images
            side (str): 'src' or 'tgt'
            truncate: maximum img size ((0,0) or None for unlimited)
        Yields:
            a dictionary containing image data, path and index for each line.
        """
        from PIL import Image
        from torchvision import transforms
        import cv2

        with codecs.open(path, "r", "utf-8") as corpus_file:
            for i, line in enumerate(corpus_file):
                filename = line.strip()
                img_path = os.path.join(src_dir, filename)
                if not os.path.exists(img_path):
                    img_path = line

                assert os.path.exists(img_path), \
                    'img path %s not found' % (line.strip())

                if channel_size == 1:
                    img = transforms.ToTensor()(
                        Image.fromarray(cv2.imread(img_path, 0)))
                else:
                    img = transforms.ToTensor()(Image.open(img_path))

                if truncate and truncate != (0, 0):
                    if not (img.size(1) <= truncate[0]
                            and img.size(2) <= truncate[1]):
                        continue

                yield {side: img, side + '_path': filename, 'indices': i}
