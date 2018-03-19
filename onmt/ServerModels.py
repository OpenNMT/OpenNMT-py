import sys
import os
import argparse
import torch
import io
import time
import codecs
import onmt
import onmt.opts
import onmt.ModelConstructor


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


class ServerModels():
    def __init__(self):
        self.models = {}
        self.next_id = 0

    def load(self, opt):
        model_id = self.next_id
        model = ServerModel(opt, load=True)
        self.models[model_id] = model
        self.next_id += 1
        return model_id, model.load_time

    def run(self, model_id, text):
        if model_id in self.models and self.models[model_id] is not None:
            return self.models[model_id].run(text)
        else:
            raise ServerModelError("No such model '%s'" % str(model_id))

    def unload(self, model_id):
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
            sys.argv += ['-%s' % k, v]

        print(sys.argv)
        opt = parser.parse_args()
        opt.cuda = opt.gpu > -1

        dummy_parser = argparse.ArgumentParser(description='train.py')
        onmt.opts.model_opts(dummy_parser)
        dummy_opt = dummy_parser.parse_known_args([])[0]

        self.opt = opt
        self.dummy_opt = dummy_opt

        sys.argv = prec_argv

    def load(self):
        self.timer.start()

        if self.opt.cuda:
            torch.cuda.set_device(self.opt.gpu)

        fields, model, model_opt = onmt.ModelConstructor.load_test_model(
            self.opt, self.dummy_opt.__dict__)

        self.fields = fields
        self.model = model
        self.model_opt = model_opt

        scorer = onmt.translate.GNMTGlobalScorer(self.opt.alpha,
                                                 self.opt.beta,
                                                 self.opt.coverage_penalty,
                                                 self.opt.length_penalty)
        self.translator = onmt.translate.Translator(
            model, fields,
            beam_size=self.opt.beam_size,
            n_best=self.opt.n_best,
            global_scorer=scorer,
            max_length=self.opt.max_length,
            copy_attn=self.model_opt.copy_attn,
            cuda=self.opt.cuda,
            beam_trace=self.opt.dump_beam != "",
            min_length=self.opt.min_length,
            stepwise_penalty=self.opt.stepwise_penalty)

        self.load_time = self.timer.tick()

    def run(self, text):
        out_file = io.StringIO()
        self.out_file = out_file

        self.timer.start()
        tmp_root = "/tmp/onmt_server"
        os.makedirs(tmp_root, exist_ok=True)
        src_path = os.path.join(tmp_root, "tmp_src")
        with codecs.open(src_path, 'w', 'utf-8') as f:
            f.write(text)
        wtime = self.timer.tick()

        data = onmt.io.build_dataset(self.fields, self.opt.data_type,
                                     src_path, None,
                                     src_dir=None,
                                     sample_rate=self.opt.sample_rate,
                                     window_size=self.opt.window_size,
                                     window_stride=self.opt.window_stride,
                                     window=self.opt.window,
                                     use_filter_pred=False)
        data_iter = onmt.io.OrderedIterator(
            dataset=data, device=self.opt.gpu,
            batch_size=self.opt.batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        builder = onmt.translate.TranslationBuilder(
            data, self.translator.fields,
            self.opt.n_best, self.opt.replace_unk, self.opt.tgt)

        # Statistics
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        for batch in data_iter:
            batch_data = self.translator.translate_batch(batch, data)
            translations = builder.from_batch(batch_data)

            for trans in translations:
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if self.opt.tgt:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.opt.n_best]]
                self.out_file.write('\n'.join(n_best_preds))
                self.out_file.write('\n')
                self.out_file.flush()

        total_time = self.timer.tick()
        times = {"writing_src": wtime,
                 "translation": total_time-wtime, "total": total_time}
        return self.out_file.getvalue(), times
