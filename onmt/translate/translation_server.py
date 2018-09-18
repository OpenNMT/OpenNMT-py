#!/usr/bin/env python
""" REST Translation server """
from __future__ import print_function
import sys
import os
import argparse
import time
import json
import threading
import re

import torch
import onmt.opts

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator


class Timer:
    def __init__(self, start=False):
        self.stime = -1
        self.prev = -1
        self.times = {}
        if start:
            self.start()

    def start(self):
        self.stime = time.time()
        self.prev = self.stime
        self.times = {}

    def tick(self, name=None, tot=False):
        t = time.time()
        if not tot:
            elapsed = t - self.prev
        else:
            elapsed = t - self.stime
        self.prev = t

        if name is not None:
            self.times[name] = elapsed
        return elapsed


class ServerModelError(Exception):
    pass


class TranslationServer():
    def __init__(self):
        self.models = {}
        self.next_id = 0

    def start(self, config_file):
        """Read the config file and pre-/load the models
        """
        self.config_file = config_file
        with open(self.config_file) as f:
            self.confs = json.load(f)

        self.models_root = self.confs.get('models_root', './available_models')
        for i, conf in enumerate(self.confs["models"]):
            if "models" not in conf:
                if "model" in conf:
                    # backwards compatibility for confs
                    conf["models"] = [conf["model"]]
                else:
                    raise ValueError("""Incorrect config file: missing 'models'
                                        parameter for model #%d""" % i)
            kwargs = {'timeout': conf.get('timeout', None),
                      'load': conf.get('load', None),
                      'tokenizer_opt': conf.get('tokenizer', None),
                      'on_timeout': conf.get('on_timeout', None),
                      'model_root': conf.get('model_root', self.models_root)
                      }
            kwargs = {k: v for (k, v) in kwargs.items() if v is not None}
            model_id = conf.get("id", None)
            opt = conf["opt"]
            opt["models"] = conf["models"]
            self.preload_model(opt, model_id=model_id, **kwargs)

    def clone_model(self, model_id, opt, timeout=-1):
        """Clone a model `model_id`.
           Different options may be passed. If `opt` is None, it will use the
           same set of options
        """
        if model_id in self.models:
            if opt is None:
                opt = self.models[model_id].user_opt
            opt["models"] = self.models[model_id].opt.models
            return self.load_model(opt, timeout)
        else:
            raise ServerModelError("No such model '%s'" % str(model_id))

    def load_model(self, opt, model_id=None, **model_kwargs):
        """Loading a model given a set of options
        """
        model_id = self.preload_model(opt, model_id=model_id, **model_kwargs)
        load_time = self.models[model_id].load_time

        return model_id, load_time

    def preload_model(self, opt, model_id=None, **model_kwargs):
        """Preloading the model: updating internal datastructure
           It will effectively load the model if `load` is set
        """
        if model_id is not None:
            if model_id in self.models.keys():
                raise ValueError("Model ID %d already exists" % model_id)
        else:
            model_id = self.next_id
            while model_id in self.models.keys():
                model_id += 1
            self.next_id = model_id + 1
        print("Pre-loading model %d" % model_id)
        model = ServerModel(opt, model_id, **model_kwargs)
        self.models[model_id] = model

        return model_id

    def run(self, inputs):
        """Translate `inputs`
           We keep the same format as the Lua version i.e.
             [{"id": model_id, "src": "sequence to translate"},{ ...}]

           We use inputs[0]["id"] as the model id
        """
        model_id = inputs[0].get("id", 0)
        if model_id in self.models and self.models[model_id] is not None:
            return self.models[model_id].run(inputs)
        else:
            print("Error No such model '%s'" % str(model_id))
            raise ServerModelError("No such model '%s'" % str(model_id))

    def unload_model(self, model_id):
        """Manually unload a model.
           It will free the memory and cancel the timer
        """
        if model_id in self.models and self.models[model_id] is not None:
            self.models[model_id].unload()
        else:
            raise ServerModelError("No such model '%s'" % str(model_id))

    def list_models(self):
        """Return the list of available models
        """
        models = []
        for _, model in self.models.items():
            models += [model.to_dict()]
        return models


class ServerModel:
    def __init__(self, opt, model_id, tokenizer_opt=None, load=False,
                 timeout=-1, on_timeout="to_cpu", model_root="./"):
        """
            Args:
                opt: (dict) options for the Translator
                model_id: (int) model id
                tokenizer_opt: (dict) options for the tokenizer or None
                load: (bool) whether to load the model during __init__
                timeout: (int) seconds before running `do_timeout`
                         Negative values means no timeout
                on_timeout: (str) in ["to_cpu", "unload"] set what to do on
                            timeout (see function `do_timeout`)
                model_root: (str) path to the model directory
                            it must contain de model and tokenizer file

        """
        self.model_root = model_root
        self.opt = self.parse_opt(opt)
        if self.opt.n_best > 1:
            raise ValueError("Values of n_best > 1 are not supported")

        self.model_id = model_id
        self.tokenizer_opt = tokenizer_opt
        self.timeout = timeout
        self.on_timeout = on_timeout

        self.unload_timer = None
        self.user_opt = opt
        self.tokenizer = None
        self.logger = init_logger(self.opt.log_file)
        self.loading_lock = threading.Event()
        self.loading_lock.set()

        if load:
            self.load()

    def parse_opt(self, opt):
        """Parse the option set passed by the user using `onmt.opts`
           Args:
               opt: (dict) options passed by the user

           Returns:
               opt: (Namespace) full set of options for the Translator
        """
        prec_argv = sys.argv
        sys.argv = sys.argv[:1]
        parser = argparse.ArgumentParser()
        onmt.opts.translate_opts(parser)

        models = opt['models']
        if not isinstance(models, (list, tuple)):
            models = [models]
        opt['models'] = [os.path.join(self.model_root, model)
                         for model in models]
        opt['src'] = "dummy_src"

        for (k, v) in opt.items():
            if k == 'models':
                sys.argv += ['-model']
                sys.argv += [str(model) for model in v]
            elif type(v) == bool:
                sys.argv += ['-%s' % k]
            else:
                sys.argv += ['-%s' % k, str(v)]

        opt = parser.parse_args()
        opt.cuda = opt.gpu > -1

        sys.argv = prec_argv
        return opt

    @property
    def loaded(self):
        return hasattr(self, 'translator')

    def load(self):
        self.loading_lock.clear()

        timer = Timer()
        self.logger.info("Loading model %d" % self.model_id)
        timer.start()

        try:
            self.translator = build_translator(self.opt,
                                               report_score=False,
                                               out_file=open(os.devnull, "w"))
        except RuntimeError as e:
            raise ServerModelError("Runtime Error: %s" % str(e))

        timer.tick("model_loading")
        if self.tokenizer_opt is not None:
            self.logger.info("Loading tokenizer")

            if "type" not in self.tokenizer_opt:
                raise ValueError(
                    "Missing mandatory tokenizer option 'type'")

            if self.tokenizer_opt['type'] == 'sentencepiece':
                if "model" not in self.tokenizer_opt:
                    raise ValueError(
                        "Missing mandatory tokenizer option 'model'")
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                model_path = os.path.join(self.model_root,
                                          self.tokenizer_opt['model'])
                sp.Load(model_path)
                self.tokenizer = sp
            elif self.tokenizer_opt['type'] == 'pyonmttok':
                if "params" not in self.tokenizer_opt:
                    raise ValueError(
                        "Missing mandatory tokenizer option 'params'")
                import pyonmttok
                if self.tokenizer_opt["mode"] is not None:
                    mode = self.tokenizer_opt["mode"]
                else:
                    mode = None
                # load can be called multiple times: modify copy
                tokenizer_params = dict(self.tokenizer_opt["params"])
                for key, value in self.tokenizer_opt["params"].items():
                    if key.endswith("path"):
                        tokenizer_params[key] = os.path.join(
                            self.model_root, value)
                tokenizer = pyonmttok.Tokenizer(mode,
                                                **tokenizer_params)
                self.tokenizer = tokenizer
            else:
                raise ValueError("Invalid value for tokenizer type")

        self.load_time = timer.tick()
        self.reset_unload_timer()
        self.loading_lock.set()

    def run(self, inputs):
        """Translate `inputs` using this model

            Args:
                inputs: [{"src": "..."},{"src": ...}]

            Returns:
                result: (list) translations
                times: (dict) containing times
        """
        self.stop_unload_timer()

        timer = Timer()
        timer.start()
        self.logger.info("Running translation using %d" % self.model_id)

        if not self.loading_lock.is_set():
            self.logger.info(
                "Model #%d is being loaded by another thread, waiting"
                % self.model_id)
            if not self.loading_lock.wait(timeout=30):
                raise ServerModelError("Model %d loading timeout"
                                       % self.model_id)

        else:
            if not self.loaded:
                self.load()
                timer.tick(name="load")
            elif self.opt.cuda:
                self.to_gpu()
                timer.tick(name="to_gpu")

        texts = []
        head_spaces = []
        tail_spaces = []
        sslength = []
        for i, inp in enumerate(inputs):
            src = inp['src']
            if src.strip() == "":
                head_spaces.append(src)
                texts.append("")
                tail_spaces.append("")
            else:
                whitespaces_before, whitespaces_after = "", ""
                match_before = re.search(r'^\s+', src)
                match_after = re.search(r'\s+$', src)
                if match_before is not None:
                    whitespaces_before = match_before.group(0)
                if match_after is not None:
                    whitespaces_after = match_after.group(0)
                head_spaces.append(whitespaces_before)
                tok = self.maybe_tokenize(src.strip())
                texts.append(tok)
                sslength.append(len(tok.split()))
                tail_spaces.append(whitespaces_after)

        empty_indices = [i for i, x in enumerate(texts) if x == ""]
        texts_to_translate = [x for x in texts if x != ""]

        scores = []
        predictions = []
        if len(texts_to_translate) > 0:
            try:
                scores, predictions = self.translator.translate(
                    src_data_iter=texts_to_translate,
                    batch_size=self.opt.batch_size)
            except RuntimeError as e:
                raise ServerModelError("Runtime Error: %s" % str(e))

        timer.tick(name="translation")
        self.logger.info("""Using model #%d\t%d inputs
               \ttranslation time: %f""" % (self.model_id, len(texts),
                                            timer.times['translation']))
        self.reset_unload_timer()

        # NOTE: translator returns lists of `n_best` list
        #       we can ignore that (i.e. flatten lists) only because
        #       we restrict `n_best=1`
        def flatten_list(_list): return sum(_list, [])
        results = flatten_list(predictions)
        scores = [score_tensor.item()
                  for score_tensor in flatten_list(scores)]

        results = [self.maybe_detokenize(item)
                   for item in results]

        # build back results with empty texts
        for i in empty_indices:
            results.insert(i, "")
            scores.insert(i, 0)

        results = ["".join(items)
                   for items in zip(head_spaces, results, tail_spaces)]

        self.logger.info("Translation Results: %d", len(results))

        return results, scores, self.opt.n_best, timer.times

    def do_timeout(self):
        """Timeout function that free GPU memory by moving the model to CPU
           or unloading it; depending on `self.on_timemout` value
        """
        if self.on_timeout == "unload":
            self.logger.info("Timeout: unloading model %d" % self.model_id)
            self.unload()
        if self.on_timeout == "to_cpu":
            self.logger.info("Timeout: sending model %d to CPU"
                             % self.model_id)
            self.to_cpu()

    def unload(self):
        self.logger.info("Unloading model %d" % self.model_id)
        del self.translator
        if self.opt.cuda:
            torch.cuda.empty_cache()
        self.unload_timer = None

    def stop_unload_timer(self):
        if self.unload_timer is not None:
            self.unload_timer.cancel()

    def reset_unload_timer(self):
        if self.timeout < 0:
            return

        self.stop_unload_timer()
        self.unload_timer = threading.Timer(self.timeout, self.do_timeout)
        self.unload_timer.start()

    def to_dict(self):
        hide_opt = ["models", "src"]
        d = {"model_id": self.model_id,
             "opt": {k: self.user_opt[k] for k in self.user_opt.keys()
                     if k not in hide_opt},
             "models": self.user_opt["models"],
             "loaded": self.loaded,
             "timeout": self.timeout,
             }
        if self.tokenizer_opt is not None:
            d["tokenizer"] = self.tokenizer_opt
        return d

    def to_cpu(self):
        """Move the model to CPU and clear CUDA cache
        """
        self.translator.model.cpu()
        if self.opt.cuda:
            torch.cuda.empty_cache()

    def to_gpu(self):
        """Move the model to GPU
        """
        torch.cuda.set_device(self.opt.gpu)
        self.translator.model.cuda()

    def maybe_tokenize(self, sequence):
        """Tokenize the sequence (or not)

           Same args/returns as `tokenize`
        """
        if self.tokenizer_opt is not None:
            return self.tokenize(sequence)
        return sequence

    def tokenize(self, sequence):
        """Tokenize a single sequence

            Args:
                sequence: (str) the sequence to tokenize

            Returns:
                tok: (str) the tokenized sequence

        """
        if self.tokenizer is None:
            raise ValueError("No tokenizer loaded")

        if self.tokenizer_opt["type"] == "sentencepiece":
            tok = self.tokenizer.EncodeAsPieces(sequence)
            tok = " ".join(tok)
        elif self.tokenizer_opt["type"] == "pyonmttok":
            tok, _ = self.tokenizer.tokenize(sequence)
            tok = " ".join(tok)
        return tok

    def maybe_detokenize(self, sequence):
        """De-tokenize the sequence (or not)

           Same args/returns as `tokenize`
        """
        if self.tokenizer_opt is not None and ''.join(sequence.split()) != '':
            return self.detokenize(sequence)
        return sequence

    def detokenize(self, sequence):
        """Detokenize a single sequence

           Same args/returns as `tokenize`
        """
        if self.tokenizer is None:
            raise ValueError("No tokenizer loaded")

        if self.tokenizer_opt["type"] == "sentencepiece":
            detok = self.tokenizer.DecodePieces(sequence.split())
        elif self.tokenizer_opt["type"] == "pyonmttok":
            detok = self.tokenizer.detokenize(sequence.split())

        return detok
