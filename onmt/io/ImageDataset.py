# -*- coding: utf-8 -*-

import os
import codecs

from onmt.io.IO import ONMTDatasetBase, _make_example, \
                    _join_dicts, _peek, _construct_example_fromlist


class ImageDataset(ONMTDatasetBase):
    """ Dataset for data_type=='img' """

    def sort_key(self, ex):
        "Sort using the size of the image."
        return (-ex.src.size(2), -ex.src.size(1))

    def _process_corpus(self, fields, src_path, src_dir, tgt_path,
                        tgt_seq_length=0, tgt_seq_length_trunc=0,
                        use_filter_pred=True):
        """
        Build Example objects, Field objects, and filter_pred function
        from image corpus.

        Args:
            fields: a dictionary of Field objects. Keys are like 'src',
                    'tgt', 'src_map', and 'alignment'.
            src_path: location of a src file containing image paths
            src_dir: location of source images
            tgt_path: location of target-side data or None.
            tgt_seq_length: maximum target sequence length.
            tgt_seq_length_trunc: truncated target sequence length.
            use_filter_pred: use a custom filter predicate to filter examples?

        Returns:
            constructed tuple of Examples objects, Field objects, filter_pred.
        """
        assert (src_dir is not None) and os.path.exists(src_dir),\
            'src_dir must be a valid directory if data_type is img'

        self.data_type = 'img'

        global Image, transforms
        from PIL import Image
        from torchvision import transforms

        # Process the source image corpus into examples, and process
        # the target text corpus into examples, if tgt_path is not None.
        src_examples = _read_img_file(src_path, src_dir, "src")
        self.n_src_feats = 0

        tgt_examples, self.n_tgt_feats = \
            _make_example(tgt_path, tgt_seq_length_trunc, "tgt")

        if tgt_examples is not None:
            examples = (_join_dicts(src, tgt)
                        for src, tgt in zip(src_examples, tgt_examples))
        else:
            examples = src_examples

        # Peek at the first to see which fields are used.
        ex, examples = _peek(examples)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples)
        out_examples = (_construct_example_fromlist(ex_values, out_fields)
                        for ex_values in example_values)

        def filter_pred(example):
            if tgt_examples is not None:
                return 0 < len(example.tgt) <= tgt_seq_length
            else:
                return True

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        return out_examples, out_fields, filter_pred


def _read_img_file(path, src_dir, side, truncate=None):
    """
    Args:
        path: location of a src file containing image paths
        src_dir: location of source images
        side: 'src' or 'tgt'
        truncate: maximum img size ((0,0) or None for unlimited)

    Yields:
        a dictionary containing image data, path and index for each line.
    """
    with codecs.open(path, "r", "utf-8") as corpus_file:
        index = 0
        for line in corpus_file:
            img_path = os.path.join(src_dir, line.strip())
            if not os.path.exists(img_path):
                img_path = line
            assert os.path.exists(img_path), \
                'img path %s not found' % (line.strip())
            img = transforms.ToTensor()(Image.open(img_path))
            if truncate and truncate != (0, 0):
                if not (img.size(1) <= truncate[0]
                        and img.size(2) <= truncate[1]):
                    continue
            example_dict = {side: img,
                            side+'_path': line.strip(),
                            'indices': index}
            index += 1
            yield example_dict
