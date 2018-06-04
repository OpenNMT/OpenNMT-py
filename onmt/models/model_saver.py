import torch
import torch.nn as nn

import onmt.inputters


def build_model_saver(model_opt, opt, model, fields, optim):
    model_saver = ModelSaver(opt.save_model,
                             model,
                             model_opt,
                             fields,
                             optim,
                             opt.keep_checkpoint)
    return model_saver


class ModelSaverBase(object):
    def __init__(self, base_path, model, model_opt, fields, optim,
                 keep_checkpoint=0):
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.optim = optim
        self.keep_checkpoint = keep_checkpoint

    def _save(self, step):
        """ Save a resumable checkpoint.

        Args:
            step (int): step number
        """
        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    def __init__(self, base_path, model, model_opt, fields, optim,
                 keep_checkpoint=0):
        super(ModelSaver, self).__init__(
            base_path, model, model_opt, fields, optim,
            keep_checkpoint=keep_checkpoint)

    def _save(self, step):
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.inputters.save_fields_to_vocab(self.fields),
            'opt': self.model_opt,
            'step': step,
            'optim': self.optim,
        }
        print("Saving checkpoint %s_step_%d.pt" % (self.base_path, step))
        torch.save(checkpoint,
                   '%s_step_%d.pt'
                   % (self.base_path, step))
