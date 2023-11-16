# -*- coding: utf-8 -*-

import torch
import random
from contextlib import contextmanager
import inspect
import numpy as np
import os
from copy import deepcopy


class RandomShuffler(object):
    """Use random functions while keeping track of the random state to make it
    reproducible and deterministic.
    taken from the torchtext Library"""

    def __init__(self, random_state=None):
        self._random_state = random_state
        if self._random_state is None:
            self._random_state = random.getstate()

    @contextmanager
    def use_internal_state(self):
        """Use a specific RNG state."""
        old_state = random.getstate()
        random.setstate(self._random_state)
        yield
        self._random_state = random.getstate()
        random.setstate(old_state)

    @property
    def random_state(self):
        return deepcopy(self._random_state)

    def __call__(self, data):
        """Shuffle and return a new list."""
        with self.use_internal_state():
            return random.sample(data, len(data))


def check_path(path, exist_ok=False, log=print):
    """Check if `path` exists, makedirs if not else warning/IOError."""
    if os.path.exists(path):
        if exist_ok:
            log(f"path {path} exists, may overwrite...")
        else:
            raise IOError(f"path {path} exists, stop.")
    else:
        if os.path.dirname(path) != "":
            os.makedirs(os.path.dirname(path), exist_ok=True)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device) >= lengths.unsqueeze(1)


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    return x.repeat_interleave(count, dim)


def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, "gpu_ranks") and len(opt.gpu_ranks) > 0) or (
        hasattr(opt, "gpu") and opt.gpu > -1
    )


def set_random_seed(seed, is_cuda):
    """Sets the random seed."""
    if seed > 0:
        torch.manual_seed(seed)
        # this one is needed for Random Shuffler of batches
        # in multi gpu it ensures datasets are read in the same order
        random.seed(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True
        # This one is needed for various tranfroms
        np.random.seed(seed)

    if is_cuda and seed > 0:
        # These ensure same initialization in multi gpu mode
        torch.cuda.manual_seed(seed)


def fn_args(fun):
    """Returns the list of function arguments name."""
    return inspect.getfullargspec(fun).args


def report_matrix(row_label, column_label, matrix):
    header_format = "{:>10.10} " + "{:>10.7} " * len(row_label)
    output = header_format.format("", *row_label) + "\n"
    for word, row in zip(column_label, matrix):
        row_format = "{:>10.10} " + "{:>10.7f} " * len(row)
        output += row_format.format(word, *row) + "\n"
    return output


def check_model_config(model_config, root):
    # we need to check the model path + any tokenizer path
    for model in model_config["models"]:
        model_path = os.path.join(root, model)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "{} from model {} does not exist".format(model_path, model_config["id"])
            )
    if "tokenizer" in model_config.keys():
        if "params" in model_config["tokenizer"].keys():
            for k, v in model_config["tokenizer"]["params"].items():
                if k.endswith("path"):
                    tok_path = os.path.join(root, v)
                    if not os.path.exists(tok_path):
                        raise FileNotFoundError(
                            "{} from model {} does not exist".format(
                                tok_path, model_config["id"]
                            )
                        )
