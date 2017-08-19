import torch.optim as optim
from torch.nn.utils import clip_grad_norm


class Optim(object):

    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.opt.optim == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.opt.optim == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.opt.optim == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.opt.optim == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr,
                                        betas=self.betas, eps=1e-9)
        else:
            raise RuntimeError("Invalid optim method: " + self.optim)

    def __init__(self, opt):
        self.last_ppl = None
        self.lr = opt.learning_rate
        self._step = 0
        self.betas = [0.9, 0.98]
        self.opt = opt
        self.start_decay = False

    def _setRate(self, lr):
        self.lr = lr
        self.optimizer.param_groups[0]['lr'] = self.lr

    def step(self):
        "Compute gradients norm."
        self._step += 1

        # Decay method used in tensor2tensor.
        if self.opt.decay_method == "noam":
            self._setRate(
                self.opt.learning_rate *
                (self.opt.rnn_size ** (-0.5) *
                 min(self._step ** (-0.5),
                     self._step * self.opt.warmup_steps**(-1.5))))

        if self.opt.max_grad_norm:
            clip_grad_norm(self.params, self.opt.max_grad_norm)
        self.optimizer.step()

    def updateLearningRate(self, ppl, epoch):
        """
        Decay learning rate if val perf does not improve
        or we hit the start_decay_at limit.
        """
        print("Learning rate:", self.lr)
        self.start_decay = False

        if self.opt.start_decay_at is not None \
           and epoch >= self.opt.start_decay_at \
           and not self.opt.decay_method == "restart":
            self.start_decay = True

        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.opt.learning_rate_decay
            print("Decaying learning rate to %g" % self.lr)

            if self.opt.decay_method == "restart":
                # Reset optim method
                # Don't update self.last_ppl (epoch resets)
                # From http://aclweb.org/anthology/W/W17/W17-3203.pdf
                self.set_parameters(self.params)
            else:
                self.last_ppl = ppl
                self.optimizer.param_groups[0]['lr'] = self.lr
        else:
            self.last_ppl = ppl
