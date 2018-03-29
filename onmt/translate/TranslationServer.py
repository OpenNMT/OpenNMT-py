import sys
import os
import argparse
import torch
import io
import time
import codecs
import json
import threading
import onmt
import onmt.opts
import onmt.translate


class Timer:
    def __init__(self, start=False):
        self.stime = -1
        if start:
            self.start()

    def start(self):
        self.stime = time.time()

    def tick(self):
        return time.time() - self.stime


class ServerModelError(Exception):
    pass


class TranslationServer():
    def __init__(self, models_root, available_models):
        self.models = {}
        self.models_root = models_root
        self.available_models = available_models

        self.next_id = 0
        with open(available_models) as f:
            self.confs = json.load(f)

        for i, conf in enumerate(self.confs["models"]):
            if "model" not in conf:
                raise ValueError("""Incorrect config file: missing 'model'
                                    parameter for model #%d""" % i)
            path = os.path.join(self.models_root, conf["model"])
            timeout = conf.get('timeout', -1)
            opt = conf.get('opt', {})
            load = conf.get('load', False)
            opt["model"] = path
            self.preload_model(opt, timeout=timeout, load=load)

    def clone_model(self, model_id, opt, timeout=-1):
        """Clone a model `model_id`.
           Different options may be passed. If `opt` is None, it will use the
           same set of options
        """
        if model_id in self.models:
            if opt is None:
                opt = self.models[model_id].user_opt
            opt["model"] = self.models[model_id].opt.model
            return self.load_model(opt, timeout)
        else:
            raise ServerModelError("No such model '%s'" % str(model_id))

    def load_model(self, opt, timeout=-1):
        """Loading a model given a set of options
        """
        model_id = self.preload_model(opt, timeout, load=True)
        load_time = self.models[model_id].load_time

        return model_id, load_time

    def preload_model(self, opt, timeout=-1, load=False):
        """Preloading the model: updating internal datastructure
           It will effectively load the model if `load` is set
        """
        model_id = self.next_id
        model = ServerModel(opt, model_id, timeout=timeout, load=load)
        self.models[model_id] = model
        self.next_id += 1
        return model_id

    def run_model(self, model_id, inputs):
        """Translate `inputs` using the model `model_id`
           Inputs must be formatted as a list of sequence
           e.g. [{"src": "..."},{"src": ...}]
        """
        if model_id in self.models and self.models[model_id] is not None:
            return self.models[model_id].run(inputs)
        else:
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
        """Lists available models
        """
        models = []

        for i, model in self.models.items():
            models += [model.toJSON()]
        return models


class ServerModel:
    def __init__(self, opt, model_id, load=False, timeout=-1):
        self.opt = self.parse_opt(opt)
        self.timer = Timer()
        self.timeout = timeout
        self.unload_timer = None
        self.user_opt = opt
        self.model_id = model_id

        if load:
            self.load()

    def parse_opt(self, opt):
        """Parse the option set passed by the user using `onmt.opts`
        """
        prec_argv = sys.argv
        parser = argparse.ArgumentParser()
        onmt.opts.translate_opts(parser)

        opt['src'] = "dummy_src"

        for (k, v) in opt.items():
            sys.argv += ['-%s' % k, str(v)]

        opt = parser.parse_args()
        opt.cuda = opt.gpu > -1

        sys.argv = prec_argv
        return opt

    @property
    def loaded(self):
        return hasattr(self, 'translator')

    def load(self):
        self.timer.start()

        self.out_file = io.StringIO()
        try:
            self.translator = onmt.translate.Translator(self.opt,
                                                        report_score=False,
                                                        out_file=self.out_file)
        except RuntimeError as e:
            raise ServerModelError("Runtime Error: %s" % str(e))

        self.load_time = self.timer.tick()
        self.reset_unload_timer()

    def run(self, inputs):
        """Translate `inputs` using this model
           Inputs must be formatted as a list of sequence
           e.g. [{"src": "..."},{"src": ...}]
        """
        print("Running translation using %d" % self.model_id)
        load_time = 0
        if not self.loaded:
            self.load()
            load_time = self.load_time

        self.timer.start()
        # NOTE: the translator exept a filepath as parameter
        #       therefore we write the data as a temp file.
        tmp_root = "/tmp/onmt_server"
        os.makedirs(tmp_root, exist_ok=True)
        src_path = os.path.join(tmp_root, "tmp_src")
        with codecs.open(src_path, 'w', 'utf-8') as f:
            for inp in inputs:
                f.write(inp['src'] + "\n")
        try:
            self.translator.translate(None, src_path, None)
        except RuntimeError as e:
            raise ServerModelError("Runtime Error: %s" % str(e))

        tr_time = self.timer.tick()
        times = {"translation": tr_time,
                 "loading": load_time,
                 "total": tr_time + load_time}
        print("Model %d, translation time: %s" % (self.model_id, str(times)))
        self.reset_unload_timer()
        result = self.out_file.getvalue().split("\n")
        return result, times

    def unload(self):
        print("Unloading model %d" % self.model_id)
        del self.translator
        if self.opt.cuda:
            torch.cuda.empty_cache()
        self.unload_timer = None

    def reset_unload_timer(self):
        if self.timeout < 0:
            return

        if self.unload_timer is not None:
            self.unload_timer.cancel()
        self.unload_timer = threading.Timer(self.timeout, self.unload)
        self.unload_timer.start()

    def toJSON(self):
        hide_opt = ["model", "src"]
        return {"model_id": self.model_id,
                "opt": {k: self.user_opt[k] for k in self.user_opt.keys()
                        if k not in hide_opt},
                "model": self.user_opt["model"],
                "loaded": self.loaded,
                "timeout": self.timeout
                }
