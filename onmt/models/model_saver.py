import os
import torch
import torch.nn as nn
from torchtext.data import Field
from collections import deque
from onmt.utils.logging import logger

from copy import deepcopy


def build_model_saver(model_opt, opt, model, fields, optim):
    model_saver = ModelSaver(opt.save_model,
                             model,
                             model_opt,
                             fields,
                             optim,
                             opt.keep_checkpoint)
    return model_saver


class ModelSaverBase(object):
    """Base class for model saving operations

    Inherited classes must implement private methods:
    * `_save`
    * `_rm_checkpoint
    """

    def __init__(self, base_path, model, model_opt, fields, optim,
                 keep_checkpoint=-1):
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.optim = optim
        self.last_saved_step = None
        self.keep_checkpoint = keep_checkpoint
        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)

    def save(self, step, moving_average=None):
        """Main entry point for model saver

        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """

        if self.keep_checkpoint == 0 or step == self.last_saved_step:
            return

        if moving_average:
            save_model = deepcopy(self.model)
            for avg, param in zip(moving_average, save_model.parameters()):
                param.data.copy_(avg.data)
        else:
            save_model = self.model

        chkpt, chkpt_name = self._save(step, save_model)
        self.last_saved_step = step

        if moving_average:
            del save_model

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_name)

    def _save(self, step):
        """Save a resumable checkpoint.

        Args:
            step (int): step number

        Returns:
            (object, str):

            * checkpoint: the saved object
            * checkpoint_name: name (or path) of the saved checkpoint
        """

        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """

        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    """Simple model saver to filesystem"""

    def _save(self, step, model):
        real_model = (model.module
                      if isinstance(model, nn.DataParallel)
                      else model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)
        if hasattr(real_model, 'bert'):
            model_state_dict = real_model.bert.state_dict()
        else:
            model_state_dict = real_model.state_dict()
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()

        # NOTE: We need to trim the vocab to remove any unk tokens that
        # were not originally here.

        vocab = deepcopy(self.fields)
        for name, field in vocab.items():
            if isinstance(field, Field):
                if hasattr(field, "vocab") and \
                   hasattr(field, "unk_token"):
                    assert name == 'tokens'
                    keys_to_pop = []
                    unk_token = field.unk_token
                    unk_id = field.vocab.stoi[unk_token]
                    for key, value in field.vocab.stoi.items():
                        if value == unk_id and key != unk_token:
                            keys_to_pop.append(key)
                    for key in keys_to_pop:
                        field.vocab.stoi.pop(key, None)
            else:
                if hasattr(field, "fields"):
                    assert name in ["src", "tgt"]
                    keys_to_pop = []
                    unk_token = field.fields[0][1].vocab.itos[0]
                    for key, value in field.fields[0][1].vocab.stoi.items():
                        if value == 0 and key != unk_token:
                            keys_to_pop.append(key)
                    for key in keys_to_pop:
                        field.fields[0][1].vocab.stoi.pop(key, None)

        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': vocab,
            'opt': self.model_opt,
            'optim': self.optim.state_dict(),
        }

        logger.info("Saving checkpoint %s_step_%d.pt" % (self.base_path, step))
        checkpoint_path = '%s_step_%d.pt' % (self.base_path, step)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        os.remove(name)
