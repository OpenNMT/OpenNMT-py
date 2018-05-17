import torch
import torch.nn as nn

import onmt.inputters


def build_model_saver(model_opt, opt, model, fields, optim):
    model_saver = ModelSaver(opt.save_model,
                             model,
                             model_opt,
                             fields,
                             optim,
                             opt.start_checkpoint_at)
    return model_saver


class ModelSaverBase(object):
    def __init__(self, base_path, model, model_opt, fields, optim, start_checkpoint_at=0):
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.optim = optim
        self.start_checkpoint_at = start_checkpoint_at

    def maybe_save(self, epoch, valid_stats):
        if epoch >= self.start_checkpoint_at:
            self._save(epoch, valid_stats)

    def force_save(self, epoch, valid_stats):
        self._save(epoch, valid_stats)

    def _save(self, epoch, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            epoch (int): epoch number
            valid_stats : statistics of last validation run
        """
        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    def __init__(self, base_path, model, model_opt, fields, optim, start_checkpoint_at=0):
        super(ModelSaver, self).__init__(
            base_path, model, model_opt, fields, optim,
            start_checkpoint_at=start_checkpoint_at)

    def _save(self, epoch, valid_stats):
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
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (self.base_path, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))
