# -*- coding: utf-8 -*-

import torch
from torchtext.data import Field

from onmt.datatypes.datatype_base import Datatype
from onmt.inputters.image_dataset import ImageDataReader


def image_sort_key(ex):
    """ Sort using the size of the image: (width, height)."""
    return ex.src.size(2), ex.src.size(1)


def batch_image(data, vocab):
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
        postprocessing=batch_image, sequential=False)
    return [], [(base_name, img)]


image = Datatype("img", ImageDataReader, image_sort_key, image_fields)
