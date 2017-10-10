from __future__ import division
import unittest
import argparse
import copy
import random

import onmt
import opts
from onmt.ModelConstructor import make_embeddings, \
                            make_encoder, make_decoder

parser = argparse.ArgumentParser(description='train.py')
opts.model_opts(parser)
opts.train_opts(parser)

# -data option is required, but not used in this test, so dummy.
opt = parser.parse_known_args(['-data', 'dummy'])[0]
print(opt)


def get_vocab():
    src = onmt.IO.ONMTDataset.get_fields()["src"]
    src.build_vocab([])
    return src.vocab


def build_model(opt):
    word_dict = get_vocab()
    feature_dicts = []

    embeddings = make_embeddings(opt, word_dict, feature_dicts)
    enc = make_encoder(opt, embeddings)

    embeddings = make_embeddings(opt, word_dict, feature_dicts,
                                 for_encoder=False)
    dec = make_decoder(opt, embeddings)

    model = onmt.Models.NMTModel(enc, dec)
    return model


class TestLearningRateUpdate(unittest.TestCase):
    """
    A unittest for testing learning rate update mechanisms.
    """
    def __init__(self, *args, **kwargs):
        super(TestLearningRateUpdate, self).__init__(*args, **kwargs)
        # This opt is from command line, we will override
        # some parameters via _add_test().
        self.opt = opt
        self.verbose = False

    def baseline_lr_step_noam(self, opt):
        """ noam update step baseline function. """
        self.step += 1

        # Decay method used in tensor2tensor.
        if opt.decay_method == "noam":
            factor = (opt.rnn_size ** (-0.5) *
                      min(self.step ** (-0.5),
                          self.step * opt.warmup_steps**(-1.5)))
            self.lr = opt.learning_rate * factor

    def baseline_epoch_step(self, ppl, epoch, opt):
        """ epoch level update baseline function. """
        if opt.decay_method == "noam":
            # If use noam, turn off other lr update method.
            return

        start_decay = False
        if epoch >= opt.start_decay_at:
            start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            start_decay = True

        if start_decay:
            self.lr *= opt.learning_rate_decay

        self.last_ppl = ppl

    def _init_settings(self, opt):
        self.num_batches = (10000 + (opt.batch_size - 1)) // opt.batch_size

        # The controller is the new lr updater to test.
        model = build_model(opt)
        self.controller = onmt.ModelController(model)

        if opt.train_from:
            # feign as we have resumed from a previous 3-epochs training.
            opt.start_epoch = 4
            self.step = (opt.start_epoch - 1) * self.num_batches - 1
            if opt.decay_method == "noam":
                self.baseline_lr_step_noam(opt)  # set the self.lr
            else:
                self.lr = random.uniform(0.37, 1.0)
            self.last_ppl = None

            self.controller.setup_optimizer(opt.optim, opt.learning_rate)
            self.controller.optimizer.state['last_lr_steps'] = self.step
            self.controller.lr = self.lr
            self.controller.setup_lr_scheduler(opt, checkpoint='dummy')
        else:
            self.step = 0
            self.lr = opt.learning_rate
            self.last_ppl = None

            self.controller.setup_optimizer(opt.optim, opt.learning_rate)
            self.controller.setup_lr_scheduler(opt)

    def _ppl_simulator(self):
        """ We simulate a decreasing ppl generator, which sometimes
        stagnates or even increases! """
        if self.last_ppl is None:
            ppl = random.uniform(1e10, 1e12)
        else:
            capricious = random.uniform(0.0, 1.0)
            if capricious - 0.9 > 1e-8:
                ppl = self.last_ppl + random.uniform(0.0, 1e4)
            else:
                ppl = random.uniform(self.last_ppl/4, self.last_ppl)

        return ppl

    def lr_update(self, opt):
        """ The test workhorse. """
        self._init_settings(opt)

        for epoch in range(opt.start_epoch, opt.epochs + 1):

            # Test noam lr update.
            for batch in range(0, self.num_batches):
                self.controller.lr_step_noam()
                self.baseline_lr_step_noam(opt)
                if opt.decay_method == 'noam' and self.verbose:
                    print("noam: step = %d, baseline lr = %.6f, lr = %.6f"
                          % (self.step, self.lr, self.controller.lr))
                self.assertEqual(self.lr, self.controller.lr)

            # Test epoch level lr update.
            ppl = self._ppl_simulator()
            self.controller.epoch_step(ppl, epoch)
            self.baseline_epoch_step(ppl, epoch, opt)
            if self.verbose:
                if self.last_ppl is None:
                    print("last_ppl = None, ppl = %.6f, baseline lr = %.6f, "
                          "lr = %.6f" % (ppl, self.lr, self.controller.lr))
                else:
                    print("last_ppl = %.6f, ppl = %.6f, baseline lr = %.6f, "
                          "lr = %.6f" % (self.last_ppl, ppl, self.lr,
                                         self.controller.lr))
            self.assertEqual(self.lr, self.controller.lr)


def _add_test(paramSetting, methodname):
    """
    Adds a Test to TestModel according to settings

    Args:
        paramSetting: list of tuples of (param, setting)
        methodname: name of the method that gets called
    """

    def test_method(self):
        if paramSetting:
            opt = copy.deepcopy(self.opt)
            for param, setting in paramSetting:
                setattr(opt, param, setting)
        else:
            opt = self.opt
        getattr(self, methodname)(opt)

    if paramSetting:
        name = 'test_' + methodname + "_" + "_".join(str(paramSetting).split())
    else:
        name = 'test_' + methodname + '_standard'

    setattr(TestLearningRateUpdate, name, test_method)
    test_method.__name__ = name


'''
TEST PARAMETERS
'''

lr_update_parameters = [
    # Non-train_from cases, use noam.
    [('decay_method', 'noam'), ('epochs', 80)],
    [('decay_method', 'noam'), ('epochs', 40), ('start_decay_at', 20)],
    [('decay_method', 'noam'), ('epochs', 40), ('learning_rate', 0.5)],
    [('decay_method', 'noam'), ('epochs', 80),
     ('batch_size', 32), ('warmup_steps', 6000)],

    # Non-train_from cases, don't use noam.
    [('epochs', 80)],
    [('epochs', 40), ('start_decay_at', 20)],
    [('epochs', 40), ('learning_rate', 0.5)],
    [('epochs', 80), ('batch_size', 32), ('warmup_steps', 6000)],

    # Below are train_from cases, we only need train_from behavior,
    # not the checkpoint, so just set it dummy.

    # train_from cases, use noam.
    [('train_from', 'dummy'), ('decay_method', 'noam'), ('epochs', 80)],
    [('train_from', 'dummy'), ('decay_method', 'noam'), ('epochs', 40),
     ('start_decay_at', 20)],
    [('train_from', 'dummy'), ('decay_method', 'noam'), ('epochs', 40),
     ('learning_rate', 0.5)],
    [('train_from', 'dummy'), ('decay_method', 'noam'), ('epochs', 80),
     ('batch_size', 32), ('warmup_steps', 6000)],

    # # train_from cases, don't use noam.
    # [('train_from', 'dummy'), ('epochs', 80)],
    # [('train_from', 'dummy'), ('epochs', 40), ('start_decay_at', 20)],
    # [('train_from', 'dummy'), ('epochs', 40), ('learning_rate', 0.5)],
    [('train_from', 'dummy'), ('epochs', 80),
     ('batch_size', 32), ('warmup_steps', 6000)],
]

for p in lr_update_parameters:
    _add_test(p, 'lr_update')
