import sys
import os
import argparse
import torch
import io
import time
import codecs
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
    def __init__(self):
        self.models = {}
        self.next_id = 0

    def load_model(self, opt):
        model_id = self.next_id
        model = ServerModel(opt, load=True)
        self.models[model_id] = model
        self.next_id += 1
        return model_id, model.load_time

    def run_model(self, model_id, text):
        if model_id in self.models and self.models[model_id] is not None:
            return self.models[model_id].run(text)
        else:
            raise ServerModelError("No such model '%s'" % str(model_id))

    def unload_model(self, model_id):
        if model_id in self.models and self.models[model_id] is not None:
            del self.models[model_id]
        else:
            raise ServerModelError("No such model '%s'" % str(model_id))


class ServerModel:
    def __init__(self, opt, load=False):
        self.set_opt(opt)
        self.timer = Timer()

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

    def load(self):
        self.timer.start()

        self.out_file = io.StringIO()
        self.translator = onmt.translate.Translator(self.opt,
                                                    report_score=False,
                                                    out_file=self.out_file)

        self.load_time = self.timer.tick()

    def run(self, text):

        self.timer.start()
        tmp_root = "/tmp/onmt_server"
        os.makedirs(tmp_root, exist_ok=True)
        src_path = os.path.join(tmp_root, "tmp_src")
        with codecs.open(src_path, 'w', 'utf-8') as f:
            f.write(text)
        wtime = self.timer.tick()

        self.translator.translate(None, src_path, None)

        total_time = self.timer.tick()
        times = {"writing_src": wtime,
                 "translation": total_time-wtime, "total": total_time}
        return self.out_file.getvalue(), times
