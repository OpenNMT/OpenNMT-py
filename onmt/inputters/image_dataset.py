# -*- coding: utf-8 -*-

import os

import torch
from torchtext.data import Field

from onmt.inputters.dataset_base import DatasetBase
from onmt.inputters.datareader_base import DataReaderBase

# domain specific dependencies
try:
    from PIL import Image
    from torchvision import transforms
    import cv2
except ImportError:
    Image, transforms, cv2 = None, None, None


class ImageDataReader(DataReaderBase):
    """
    Args:
        truncate: maximum img size ((0,0) or None for unlimited)
        channel_size: Number of channels per image.
    """

    def __init__(self, truncate=None, channel_size=3):
        self._check_deps()
        self.truncate = truncate
        self.channel_size = channel_size

    @classmethod
    def from_opt(cls, opt):
        return cls(channel_size=opt.image_channel_size)

    @classmethod
    def _check_deps(cls):
        if any([Image is None, transforms is None, cv2 is None]):
            cls._raise_missing_dep(
                "PIL", "torchvision", "cv2")

    def read(self, images, side, img_dir=None):
        """
        Args:
            images (str): location of a src file containing image paths
            src_dir (str): location of source images
            side (str): 'src' or 'tgt'
        Yields:
            a dictionary containing image data, path and index for each line.
        """
        if isinstance(images, str):
            images = DataReaderBase._read_file(images)

        for i, filename in enumerate(images):
            filename = filename.decode("utf-8").strip()
            img_path = os.path.join(img_dir, filename)
            if not os.path.exists(img_path):
                img_path = filename

            assert os.path.exists(img_path), \
                'img path %s not found' % filename

            if self.channel_size == 1:
                img = transforms.ToTensor()(
                    Image.fromarray(cv2.imread(img_path, 0)))
            else:
                img = transforms.ToTensor()(Image.open(img_path))
            if self.truncate and self.truncate != (0, 0):
                if not (img.size(1) <= self.truncate[0]
                        and img.size(2) <= self.truncate[1]):
                    continue
            yield {side: img, side + '_path': filename, 'indices': i}


class ImageDataset(DatasetBase):
    @staticmethod
    def sort_key(ex):
        """ Sort using the size of the image: (width, height)."""
        return ex.src.size(2), ex.src.size(1)


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
