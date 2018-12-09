# -*- coding: utf-8 -*-

import codecs
import os

from onmt.inputters.dataset_base import DatasetBase


class ImageDataset(DatasetBase):
    """
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
    @staticmethod
    def sort_key(ex):
        """ Sort using the size of the image: (width, height)."""
        return ex.src.size(2), ex.src.size(1)

    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 num_src_feats=0, num_tgt_feats=0,
                 filter_pred=None, image_channel_size=3):
        self.data_type = 'img'

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats

        self.image_channel_size = image_channel_size
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
        examples = [self._construct_example_fromlist(ex_values, fields)
                    for ex_values in example_values]

        super(ImageDataset, self).__init__(examples, fields, filter_pred)

    @staticmethod
    def make_image_examples_nfeats_tpl(img_iter, img_path, img_dir,
                                       image_channel_size=3):
        """
        Note: one of img_iter and img_path must be not None
        Args:
            img_iter(iterator): an iterator that yields pairs (img, filename)
                (or None)
            img_path(str): location of a src file containing image paths
                (or None)
            src_dir (str): location of source images

        Returns:
            (example_dict iterator, num_feats) tuple
        """
        if img_iter is None and img_path is None:
            raise ValueError("Either img_iter or img_path must be non-None")
        if img_iter is None:
            img_iter = ImageDataset.make_img_iterator_from_file(
                img_path, img_dir, image_channel_size
            )

        examples_iter = ImageDataset.make_examples(img_iter, img_dir, 'src')

        return examples_iter, 0

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
    def make_img_iterator_from_file(path, src_dir, image_channel_size=3):
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

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        For image corpus, source side is in form of image, thus
        no feature; while target side is in form of text, thus
        we can extract its text features.

        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.

        Returns:
            number of features on `side`.
        """
        if side == 'src':
            return 0
        with codecs.open(corpus_file, "r", "utf-8") as cf:
            f_line = cf.readline().strip().split()
            _, _, num_feats = ImageDataset.extract_text_features(f_line)
            return num_feats
