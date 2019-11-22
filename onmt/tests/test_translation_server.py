import unittest
from onmt.translate.translation_server import ServerModel, TranslationServer

import os
from six import string_types
from textwrap import dedent

import torch

from onmt.translate.translator import Translator


TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class TestServerModel(unittest.TestCase):
    def test_deferred_loading_model_and_unload(self):
        model_id = 0
        opt = {"models": ["test_model.pt"]}
        model_root = TEST_DIR
        sm = ServerModel(opt, model_id, model_root=model_root, load=False)
        self.assertFalse(sm.loaded)
        sm.load()
        self.assertTrue(sm.loaded)
        self.assertIsInstance(sm.translator, Translator)
        sm.unload()
        self.assertFalse(sm.loaded)

    def test_load_model_on_init_and_unload(self):
        model_id = 0
        opt = {"models": ["test_model.pt"]}
        model_root = TEST_DIR
        sm = ServerModel(opt, model_id, model_root=model_root, load=True)
        self.assertTrue(sm.loaded)
        self.assertIsInstance(sm.translator, Translator)
        sm.unload()
        self.assertFalse(sm.loaded)

    def test_tokenizing_with_no_tokenizer_fails(self):
        model_id = 0
        opt = {"models": ["test_model.pt"]}
        model_root = TEST_DIR
        sm = ServerModel(opt, model_id, model_root=model_root, load=True)
        with self.assertRaises(ValueError):
            sm.tokenize("hello world")

    def test_detokenizing_with_no_tokenizer_fails(self):
        model_id = 0
        opt = {"models": ["test_model.pt"]}
        model_root = TEST_DIR
        sm = ServerModel(opt, model_id, model_root=model_root, load=True)
        with self.assertRaises(ValueError):
            sm.detokenize("hello world")

    if torch.cuda.is_available():
        def test_moving_to_gpu_and_back(self):
            torch.cuda.set_device(torch.device("cuda", 0))
            model_id = 0
            opt = {"models": ["test_model.pt"]}
            model_root = TEST_DIR
            sm = ServerModel(opt, model_id, model_root=model_root, load=True)
            for p in sm.translator.model.parameters():
                self.assertEqual(p.device.type, "cpu")
            sm.to_gpu()
            for p in sm.translator.model.parameters():
                self.assertEqual(p.device.type, "cuda")
                self.assertEqual(p.device.index, 0)
            sm.to_cpu()
            for p in sm.translator.model.parameters():
                self.assertEqual(p.device.type, "cpu")

        def test_initialize_on_gpu_and_move_back(self):
            torch.cuda.set_device(torch.device("cuda", 0))
            model_id = 0
            opt = {"models": ["test_model.pt"], "gpu": 0}
            model_root = TEST_DIR
            sm = ServerModel(opt, model_id, model_root=model_root, load=True)
            for p in sm.translator.model.parameters():
                self.assertEqual(p.device.type, "cuda")
                self.assertEqual(p.device.index, 0)
            sm.to_gpu()
            for p in sm.translator.model.parameters():
                self.assertEqual(p.device.type, "cuda")
                self.assertEqual(p.device.index, 0)
            sm.to_cpu()
            for p in sm.translator.model.parameters():
                self.assertEqual(p.device.type, "cpu")

        if torch.cuda.device_count() > 1:
            def test_initialize_on_nonzero_gpu_and_back(self):
                torch.cuda.set_device(torch.device("cuda", 1))
                model_id = 0
                opt = {"models": ["test_model.pt"], "gpu": 1}
                model_root = TEST_DIR
                sm = ServerModel(opt, model_id, model_root=model_root,
                                 load=True)
                for p in sm.translator.model.parameters():
                    self.assertEqual(p.device.type, "cuda")
                    self.assertEqual(p.device.index, 1)
                sm.to_gpu()
                for p in sm.translator.model.parameters():
                    self.assertEqual(p.device.type, "cuda")
                    self.assertEqual(p.device.index, 1)
                sm.to_cpu()
                for p in sm.translator.model.parameters():
                    self.assertEqual(p.device.type, "cpu")

    def test_run(self):
        model_id = 0
        opt = {"models": ["test_model.pt"]}
        model_root = TEST_DIR
        sm = ServerModel(opt, model_id, model_root=model_root, load=True)
        inp = [{"src": "hello how are you today"},
               {"src": "good morning to you ."}]
        results, scores, n_best, time, aligns = sm.run(inp)
        self.assertIsInstance(results, list)
        for sentence_string in results:
            self.assertIsInstance(sentence_string, string_types)
        self.assertIsInstance(scores, list)
        for elem in scores:
            self.assertIsInstance(elem, float)
        self.assertIsInstance(aligns, list)
        for align_string in aligns:
            if align_string is not None:
                self.assertIsInstance(align_string, string_types)
        self.assertEqual(len(results), len(scores))
        self.assertEqual(len(scores), len(inp) * n_best)
        self.assertEqual(len(time), 1)
        self.assertIsInstance(time, dict)
        self.assertIn("translation", time)


class TestTranslationServer(unittest.TestCase):
    # this could be considered an integration test because it touches
    # the filesystem for the config file (and the models)

    CFG_F = os.path.join(
        TEST_DIR, "test_translation_server_config_file.json")

    def tearDown(self):
        if os.path.exists(self.CFG_F):
            os.remove(self.CFG_F)

    def write(self, cfg):
        with open(self.CFG_F, "w") as f:
            f.write(cfg)

    CFG_NO_LOAD = dedent("""\
        {
            "models_root": "%s",
            "models": [
                {
                    "id": 100,
                    "model": "test_model.pt",
                    "timeout": -1,
                    "on_timeout": "to_cpu",
                    "load": false,
                    "opt": {
                        "beam_size": 5
                    }
                }
            ]
        }
        """ % TEST_DIR)

    def test_start_without_initial_loading(self):
        self.write(self.CFG_NO_LOAD)
        sv = TranslationServer()
        sv.start(self.CFG_F)
        self.assertFalse(sv.models[100].loaded)
        self.assertEqual(set(sv.models.keys()), {100})

    CFG_LOAD = dedent("""\
        {
            "models_root": "%s",
            "models": [
                {
                    "id": 100,
                    "model": "test_model.pt",
                    "timeout": -1,
                    "on_timeout": "to_cpu",
                    "load": true,
                    "opt": {
                        "beam_size": 5
                    }
                }
            ]
        }
        """ % TEST_DIR)

    def test_start_with_initial_loading(self):
        self.write(self.CFG_LOAD)
        sv = TranslationServer()
        sv.start(self.CFG_F)
        self.assertTrue(sv.models[100].loaded)
        self.assertEqual(set(sv.models.keys()), {100})

    CFG_2_MODELS = dedent("""\
        {
            "models_root": "%s",
            "models": [
                {
                    "id": 100,
                    "model": "test_model.pt",
                    "timeout": -1,
                    "on_timeout": "to_cpu",
                    "load": true,
                    "opt": {
                        "beam_size": 5
                    }
                },
                {
                    "id": 1000,
                    "model": "test_model2.pt",
                    "timeout": -1,
                    "on_timeout": "to_cpu",
                    "load": false,
                    "opt": {
                        "beam_size": 5
                    }
                }
            ]
        }
        """ % TEST_DIR)

    def test_start_with_two_models(self):
        self.write(self.CFG_2_MODELS)
        sv = TranslationServer()
        sv.start(self.CFG_F)
        self.assertTrue(sv.models[100].loaded)
        self.assertFalse(sv.models[1000].loaded)
        self.assertEqual(set(sv.models.keys()), {100, 1000})
