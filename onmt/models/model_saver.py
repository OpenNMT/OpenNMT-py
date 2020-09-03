import os
import torch

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

    def save(self, step, moving_average=None, best_step=None,
             validation_ppl=None):
        """Main entry point for model saver

        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """

        if self.keep_checkpoint == 0 or step == self.last_saved_step:
            return

        save_model = self.model
        if moving_average:
            model_params_data = []
            for avg, param in zip(moving_average, save_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data

        chkpt, chkpt_name = self._save(step, save_model)
        self.last_saved_step = step

        if moving_average:
            for param_data, param in zip(model_params_data,
                                         save_model.parameters()):
                param.data = param_data

        if self.keep_checkpoint > 0:
            best_step, is_best = self._get_best_checkpoint(best_step,
                                                           validation_ppl)
            if is_best:
                if best_step is None:
                    best_checkpoint = '%s_step_%d.pt' % (self.base_path, step)
                else:
                    best_checkpoint = '%s_step_%d.pt' \
                                        % (self.base_path, best_step)
                self._update_best_config(step, validation_ppl)
            else:
                best_checkpoint = '%s_step_%d.pt' % (self.base_path, best_step)
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                if todel != best_checkpoint:
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
        model_state_dict = model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = model.generator.state_dict()

        # NOTE: We need to trim the vocab to remove any unk tokens that
        # were not originally here.

        vocab = deepcopy(self.fields)
        for side in ["src", "tgt"]:
            keys_to_pop = []
            if hasattr(vocab[side], "fields"):
                unk_token = vocab[side].fields[0][1].vocab.itos[0]
                for key, value in vocab[side].fields[0][1].vocab.stoi.items():
                    if value == 0 and key != unk_token:
                        keys_to_pop.append(key)
                for key in keys_to_pop:
                    vocab[side].fields[0][1].vocab.stoi.pop(key, None)

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
        if os.path.exists(name):
            os.remove(name)

    def _get_best_checkpoint(self, best_step, validation_ppl):
        import json
        is_best = False
        best_ckpt_config_file = self.base_path + 'best_ckpt_config.json'
        if os.path.exists(best_ckpt_config_file):
            with open(best_ckpt_config_file, 'r') as best_config:
                best_ckpt_dict = json.load(best_config)
            if validation_ppl < best_ckpt_dict['validation_ppl']:
                is_best = True
            else:
                best_step = best_ckpt_dict['step']
        else:
            is_best = True

        return best_step, is_best

    def _update_best_config(self, step, validation_ppl):
        import json
        best_ckpt_config_file = self.base_path + 'best_ckpt_config.json'
        with open(best_ckpt_config_file, 'w') as best_config:
            best_ckpt_dict = {'step': step, 'validation_ppl': validation_ppl}
            json.dump(best_ckpt_dict, best_config)
