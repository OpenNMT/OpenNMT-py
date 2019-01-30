# -*- coding: utf-8 -*-

import os

import torch
from torchtext.data import Field

from onmt.inputters.dataset_base import DatasetBase

# domain specific dependencies
try:
    from PIL import Image
    from torchvision import transforms
    import cv2
except ImportError:
    Image, transforms, cv2 = None, None, None


class ImageDataset(DatasetBase):
    @staticmethod
    def _check_deps():
        if any([Image is None, transforms is None, cv2 is None]):
            ImageDataset._raise_missing_dep(
                "PIL", "torchvision", "cv2")

    @staticmethod
    def sort_key(ex):
        """ Sort using the size of the image: (width, height)."""
        return ex.src.size(2), ex.src.size(1)

    @classmethod
    def make_examples(
        cls, images, src_dir, side, truncate=None, channel_size=3
    ):
        """
        Args:
            path (str): location of a src file containing image paths
            src_dir (str): location of source images
            side (str): 'src' or 'tgt'
            truncate: maximum img size ((0,0) or None for unlimited)
        Yields:
            a dictionary containing image data, path and index for each line.
        """
        ImageDataset._check_deps()

        if isinstance(images, str):
            images = cls._read_file(images)

        for i, filename in enumerate(images):
            filename = filename.decode("utf-8").strip()
            img_path = os.path.join(src_dir, filename)
            if not os.path.exists(img_path):
                img_path = filename

            assert os.path.exists(img_path), \
                'img path %s not found' % filename

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


def batch_img(data, vocab):
    c = data[0].size(0)
    h = max([t.size(1) for t in data])
    w = max([t.size(2) for t in data])
    imgs = torch.zeros(len(data), c, h, w).fill_(1)
    for i, img in enumerate(data):
        imgs[i, :, 0:img.size(1), 0:img.size(2)] = img
    return imgs


def image_fields(base_name, **kwargs):
    img = Field(
        use_vocab=False, dtype=torch.float,
        postprocessing=batch_img, sequential=False)
    return [(base_name, img)]
