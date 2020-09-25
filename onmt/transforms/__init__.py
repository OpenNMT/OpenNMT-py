"""Module for dynamic data transfrom."""
import os
import importlib

from .transform import make_transforms, get_specials,\
    save_transforms, load_transforms, TransformPipe,\
    Transform


AVAILABLE_TRANSFORMS = {}


def get_transforms_cls(transform_names):
    """Return valid transform class indicated in `transform_names`."""
    transforms_cls = {}
    for name in transform_names:
        if name not in AVAILABLE_TRANSFORMS:
            raise ValueError("specified tranform not supported!")
        transforms_cls[name] = AVAILABLE_TRANSFORMS[name]
    return transforms_cls


__all__ = ["get_transforms_cls", "get_specials", "make_transforms",
           "load_transforms", "save_transforms", "TransformPipe"]


def register_transform(name):
    """Transform register that can be used to add new transform class."""

    def register_transfrom_cls(cls):
        if name in AVAILABLE_TRANSFORMS:
            raise ValueError(
                'Cannot register duplicate transform ({})'.format(name))
        if not issubclass(cls, Transform):
            raise ValueError('transform ({}: {}) must extend Transform'.format(
                name, cls.__name__))
        AVAILABLE_TRANSFORMS[name] = cls
        return cls

    return register_transfrom_cls


# Auto import python files in this directory
transform_dir = os.path.dirname(__file__)
for file in os.listdir(transform_dir):
    path = os.path.join(transform_dir, file)
    if not file.startswith('_') and not file.startswith('.') and (
            file.endswith('.py') or os.path.isdir(path)):
        file_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(
            'onmt.transforms.' + file_name)
