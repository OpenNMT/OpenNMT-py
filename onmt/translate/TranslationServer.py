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

    def load_model(self, opt, timeout=-1):
        model_id = self.preload_model(opt, timeout, load=True)
        load_time = self.models[model_id].load_time

        return model_id, load_time

    def preload_model(self, opt, timeout=-1, load=False):
        model_id = self.next_id
        model = ServerModel(opt, timeout=timeout, load=load)
        self.models[model_id] = model
        self.next_id += 1
        return model_id

    def run_model(self, model_id, inputs):
        if model_id in self.models and self.models[model_id] is not None:
            return self.models[model_id].run(inputs)
        else:
            raise ServerModelError("No such model '%s'" % str(model_id))

    def unload_model(self, model_id):
        if model_id in self.models and self.models[model_id] is not None:
            self.models[model_id].unload()
        else:
            raise ServerModelError("No such model '%s'" % str(model_id))

    def list_models(self):
        models = []
        for i, model in enumerate(self.confs["models"]):
            model["loaded"] = self.models[i].loaded
            models += [model]
        return models


class ServerModel:
    def __init__(self, opt, load=False, timeout=-1):
        self.set_opt(opt)
        self.timer = Timer()
        self.timeout = timeout
        self.unload_timer = None
        if load:
            self.load()

    def set_opt(self, opt):
        prec_argv = sys.argv
        parser = argparse.ArgumentParser()
        onmt.opts.translate_opts(parser)

        opt['src'] = "dummy_src"

        for (k, v) in opt.items():
            sys.argv += ['-%s' % k, str(v)]

        opt = parser.parse_args()
        opt.cuda = opt.gpu > -1

        self.opt = opt
        sys.argv = prec_argv

    @property
    def loaded(self):
        return hasattr(self, 'translator')

    def load(self):
        self.timer.start()

        self.out_file = io.StringIO()
        self.translator = onmt.translate.Translator(self.opt,
                                                    report_score=False,
                                                    out_file=self.out_file)

        self.load_time = self.timer.tick()
        self.reset_unload_timer()

    def run(self, inputs):
        load_time = 0
        if not self.loaded:
            self.load()
            load_time = self.load_time

        self.timer.start()
        tmp_root = "/tmp/onmt_server"
        os.makedirs(tmp_root, exist_ok=True)
        src_path = os.path.join(tmp_root, "tmp_src")
        with codecs.open(src_path, 'w', 'utf-8') as f:
            for inp in inputs:
                f.write(inp['src'] + "\n")

        self.translator.translate(None, src_path, None)

        tr_time = self.timer.tick()
        times = {"translation": tr_time,
                 "loading": load_time,
                 "total": tr_time + load_time}

        self.reset_unload_timer()
        result = self.out_file.getvalue().split("\n")
        return result, times

    def unload(self):
        print("Unloading model")
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
        print("reset unload timer")
