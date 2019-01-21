# -*- coding: utf-8 -*-

import os

from onmt.datatypes.datareader_base import DataReaderBase

# imports of datatype-specific dependencies
try:
    from PIL import Image
    from torchvision import transforms
    import cv2
except ImportError:
    Image, transforms, cv2 = None, None, None


class ImageDataReader(DataReaderBase):
    """Create an image dataset reader.

    Args:
        truncate: maximum img size given as (rows, cols)
            ((0,0) or None for unlimited)
        channel_size (int): Number of image channels. Set to 1 for
            grayscale.
    """

    def __init__(self, truncate=None, channel_size=3):
        super(ImageDataReader, self).__init__()
        self._check_deps()
        self.truncate = truncate
        self.channel_size = channel_size

    @classmethod
    def from_opt(cls, opt):
        """Alternative constructor."""
        return cls(channel_size=opt.image_channel_size)

    @staticmethod
    def _check_deps():
        if any([Image is None, transforms is None, cv2 is None]):
            ImageDataReader._raise_missing_dep(
                "PIL", "torchvision", "cv2")

    @staticmethod
    def sort_key(ex):
        """ Sort using the size of the image: (width, height)."""
        return ex.src.size(2), ex.src.size(1)

    def read(self, im_files, side, src_dir=None):
        """Read images.

        Args:
            im_files (str, List[str]): Either a file of one file path per
                line (either existing path or path relative to `src_dir`)
                or a list thereof.
            src_dir (str): location of source images
            side (str): 'src' or 'tgt'

        Yields:
            a dictionary containing image data, path and index for each line.
        """

        if isinstance(im_files, str):
            im_files = self._read_file(im_files)

        for i, line in enumerate(im_files):
            filename = line.strip()
            img_path = os.path.join(src_dir, filename)
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
