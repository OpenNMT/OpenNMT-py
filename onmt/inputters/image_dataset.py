# -*- coding: utf-8 -*-
"""
    ImageDataset
"""

import codecs
import os

import torch
import torchtext

from onmt.inputters.dataset_base import DatasetBase, PAD_WORD, BOS_WORD, \
    EOS_WORD


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

    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 num_src_feats=0, num_tgt_feats=0,
                 tgt_seq_length=0, use_filter_pred=True,
                 image_channel_size=3):
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

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)
        out_examples = (self._construct_example_fromlist(
            ex_values, out_fields)
            for ex_values in example_values)
        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        out_examples = list(out_examples)

        def filter_pred(example):
            """ ? """
            if tgt_examples_iter is not None:
                return 0 < len(example.tgt) <= tgt_seq_length
            else:
                return True

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(ImageDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    def sort_key(self, ex):
        """ Sort using the size of the image: (width, height)."""
        return (ex.src.size(2), ex.src.size(1))

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
        if img_iter is None:
            if img_path is not None:
                img_iter = ImageDataset. \
                    make_img_iterator_from_file(img_path,
                                                img_dir,
                                                image_channel_size)
            else:
                raise ValueError("""One of 'img_iter' and 'img_path'
                                    must be not None""")
        examples_iter = ImageDataset.make_examples(img_iter, img_dir, 'src')
        num_feats = 0  # Source side(img) has no features.

        return (examples_iter, num_feats)

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
        assert (src_dir is not None) and os.path.exists(src_dir), \
            'src_dir must be a valid directory if data_type is img'

        for index, (img, filename) in enumerate(img_iter):
            if truncate and truncate != (0, 0):
                if not (img.size(1) <= truncate[0]
                        and img.size(2) <= truncate[1]):
                    continue

            example_dict = {side: img,
                            side + '_path': filename,
                            'indices': index}
            yield example_dict

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

                if (image_channel_size == 1):
                    img = transforms.ToTensor()(
                        Image.fromarray(cv2.imread(img_path, 0)))
                else:
                    img = transforms.ToTensor()(Image.open(img_path))

                yield img, filename

    @staticmethod
    def get_fields(n_src_features, n_tgt_features):
        """
        Args:
            n_src_features: the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features: the number of target features to
                create `torchtext.data.Field` for.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        def make_img(data, vocab):
            """ ? """
            c = data[0].size(0)
            h = max([t.size(1) for t in data])
            w = max([t.size(2) for t in data])
            imgs = torch.zeros(len(data), c, h, w).fill_(1)
            for i, img in enumerate(data):
                imgs[i, :, 0:img.size(1), 0:img.size(2)] = img
            return imgs

        fields["src"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_img, sequential=False)

        for j in range(n_src_features):
            fields["src_feat_" + str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        for j in range(n_tgt_features):
            fields["tgt_feat_" + str(j)] = \
                torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD,
                                     pad_token=PAD_WORD)

        def make_src(data, vocab):
            """ ? """
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(src_size, len(data), src_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)

        def make_tgt(data, vocab):
            """ ? """
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[:sent.size(0), i] = sent
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            sequential=False)

        return fields

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
            num_feats = 0
        else:
            with codecs.open(corpus_file, "r", "utf-8") as cf:
                f_line = cf.readline().strip().split()
                _, _, num_feats = ImageDataset.extract_text_features(f_line)

        return num_feats
