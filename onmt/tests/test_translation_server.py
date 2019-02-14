import unittest
from onmt.translate.translation_server import ServerModel

import os

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
        results, scores, n_best, time = sm.run(inp)
        self.assertIsInstance(results, list)
        for sentence_string in results:
            self.assertIsInstance(sentence_string, str)
        self.assertIsInstance(scores, list)
        for elem in scores:
            self.assertIsInstance(elem, float)
        self.assertEqual(len(results), len(scores))
        self.assertEqual(len(scores), len(inp))
        self.assertEqual(n_best, 1)
        self.assertEqual(len(time), 1)
        self.assertIsInstance(time, dict)
        self.assertIn("translation", time)

    def test_nbest_init_fails(self):
        model_id = 0
        opt = {"models": ["test_model.pt"], "n_best": 2}
        model_root = TEST_DIR
        with self.assertRaises(ValueError):
            sm = ServerModel(opt, model_id, model_root=model_root, load=True)
