from __future__ import division
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm


class ModelController(object):
    """
    This class contains stuff that is in charge of the dynamic control of
    the seq2seq model, i.e. the optimizer, the learning rate update, and
    the grad norm normalization, etc.
    """
    def __init__(self, model):
        self.model = model

    def setup_optimizer(self, optimizer, initial_lr, checkpoint=None):
        """ Setup a torch.optim.Optimizer. """
        self.optimizer = self._make_optimizer(optimizer, initial_lr)

        if checkpoint:
            print('Loading optimizer from checkpoint.')
            dummy_optimizer = self._make_optimizer(optimizer, initial_lr)
            dummy_optimizer.load_state_dict(checkpoint['optimizer'])
            last_lr_steps = dummy_optimizer.state.get('last_lr_steps', 0)
            if not last_lr_steps == 0:
                # For resuming noam steps.
                self.optimizer.state['last_lr_steps'] = last_lr_steps

    def _make_optimizer(self, optimizer, initial_lr):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if optimizer == 'sgd':
            optimizer = optim.SGD(params, lr=initial_lr)
        elif optimizer == 'adagrad':
            optimizer = optim.Adagrad(params, lr=initial_lr)
        elif optimizer == 'adadelta':
            optimizer = optim.Adadelta(params, lr=initial_lr)
        elif optimizer == 'adam':
            optimizer = optim.Adam(params, lr=initial_lr,
                                   betas=[0.9, 0.98], eps=1e-9)
        else:
            raise RuntimeError("Invalid optim method: " + optimizer)
        return optimizer

    def setup_lr_scheduler(self, opt, checkpoint=None):
        """ Setup lr_schedulers for updating lr. """
        assert self.optimizer is not None

        if opt.decay_method == 'noam':
            # 1. Batch level lr update: using 'noam' method.
            def batch_lr(step):
                factor = opt.rnn_size ** (-0.5) * \
                    min(step ** (-0.5),
                        step * opt.warmup_steps ** (-1.5))
                return factor

            # For resuming to last batch-level lr update step.
            last_lr_steps = self.optimizer.state['last_lr_steps'] \
                if checkpoint is not None else 0

            # LambdaLR requires 'initial_lr' if last_epoch is passed in.
            self.optimizer.param_groups[0]['initial_lr'] = self.lr

            self.lr_noam_scheduler = lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=batch_lr, last_epoch=last_lr_steps)

            if checkpoint is not None:
                # LambdaLR updates based on its stored base_lr, so update it.
                initial_lr = opt.learning_rate
                self.lr_noam_scheduler.base_lrs = [initial_lr]

            # When noam is used, we just turn off the other two, because noam
            # updates lr based on initial lr, which effectively ignores other
            # update methods.
        else:
            # 2. Epoch level lr update: starting from epoch 'start_decay_at'
            #    onward. lr_scheduler is cubersome for this, we do it manually.
            self.lr_decay = opt.learning_rate_decay
            self.start_decay_at = opt.start_decay_at

            # 3. Epoch level lr update: only when perplexity increases at this
            #    epoch. A negative threshold to allow equalness.
            self.lr_ppl_scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min', factor=self.lr_decay,
                threshold=-1e-4, threshold_mode='abs',
                patience=0, verbose=True)

    def setup_grad_norm_clip(self, max_grad_norm):
        self.max_grad_norm = max_grad_norm

    def lr_step_noam(self, verbose=True):
        """ lr update step for noam decay logic."""
        if getattr(self, 'lr_noam_scheduler', None) is None:
            return

        # lr_noam_scheduler is based on initial lr(its stored base_lr).
        self.lr_noam_scheduler.step()
        step = self.lr_noam_scheduler.last_epoch
        if verbose:
            print("Step %d: noam update, new lr: %.6f" % (step, self.lr))

        # For resuming from train_from.
        self.optimizer.state['last_lr_steps'] = step

    def lr_step_start_decay_at(self, epoch, verbose=True):
        """ lr update step for the 'start_decay_at' logic. """
        if epoch >= self.start_decay_at:
            self.lr = self.lr * self.lr_decay
            if verbose:
                print("Epoch %d: start_decay_at threshold update, "
                      "new lr: %.6f" % (epoch, self.lr))

    def lr_step_ppl(self, ppl, epoch, verbose=True):
        """ lr update step for 'ppl doesn't decrease this epoch' logic. """
        if getattr(self, 'lr_ppl_scheduler', None) is None:
            return

        old_lr = self.lr
        self.lr_ppl_scheduler.step(ppl, epoch)
        if verbose and not self.lr == old_lr:
            print("Epoch %d: update lr because ppl doesn't decrease this "
                  "epoch, new lr: %.6f" % (epoch, self.lr))

    def grad_norm_clip(self):
        """ Conditionally clip gradient norm. """
        if self.max_grad_norm:
            for group in self.optimizer.param_groups:
                clip_grad_norm(group['params'], self.max_grad_norm)

    def epoch_step(self, ppl, epoch):
        """ Epoch level lr update. """
        old_lr = self.lr

        self.lr_step_start_decay_at(epoch)

        # Only update once in one epoch.
        if np.isclose([self.lr], [old_lr]).all():
            self.lr_step_ppl(ppl, epoch)

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    @lr.setter
    def lr(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr
