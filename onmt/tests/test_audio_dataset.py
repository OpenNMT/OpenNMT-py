# -*- coding: utf-8 -*-
import unittest
from onmt.inputters.audio_dataset import AudioSeqField, AudioDataReader

import itertools
import os
import shutil

import torch
import torchaudio

from onmt.tests.utils_for_tests import product_dict


class TestAudioField(unittest.TestCase):
    INIT_CASES = list(product_dict(
        pad_index=[0, 32],
        batch_first=[False, True],
        include_lengths=[True, False]))

    PARAMS = list(product_dict(
        batch_size=[1, 17],
        max_len=[23],
        full_length_seq=[0, 5, 16],
        nfeats=[1, 5]))

    @classmethod
    def degenerate_case(cls, init_case, params):
        if params["batch_size"] < params["full_length_seq"]:
            return True
        return False

    @classmethod
    def pad_inputs(cls, params):
        lengths = torch.randint(1, params["max_len"],
                                (params["batch_size"],)).tolist()
        lengths[params["full_length_seq"]] = params["max_len"]
        fake_input = [
            torch.randn((params["nfeats"], lengths[b]))
            for b in range(params["batch_size"])]
        return fake_input, lengths

    @classmethod
    def numericalize_inputs(cls, init_case, params):
        bs = params["batch_size"]
        max_len = params["max_len"]
        lengths = torch.randint(1, max_len, (bs,))
        lengths[params["full_length_seq"]] = max_len
        nfeats = params["nfeats"]
        fake_input = torch.full(
            (bs, 1, nfeats, max_len), init_case["pad_index"])
        for b in range(bs):
            fake_input[b, :, :, :lengths[b]] = torch.randn(
                (1, nfeats, lengths[b]))
        if init_case["include_lengths"]:
            fake_input = (fake_input, lengths)
        return fake_input, lengths

    def test_pad_shape_and_lengths(self):
        for init_case, params in itertools.product(
                self.INIT_CASES, self.PARAMS):
            if not self.degenerate_case(init_case, params):
                field = AudioSeqField(**init_case)
                fake_input, lengths = self.pad_inputs(params)
                outp = field.pad(fake_input)
                if init_case["include_lengths"]:
                    outp, _ = outp
                expected_shape = (
                    params["batch_size"], 1, params["nfeats"],
                    params["max_len"])
                self.assertEqual(outp.shape, expected_shape)

    def test_pad_returns_correct_lengths(self):
        for init_case, params in itertools.product(
                self.INIT_CASES, self.PARAMS):
            if not self.degenerate_case(init_case, params) and \
                    init_case["include_lengths"]:
                field = AudioSeqField(**init_case)
                fake_input, lengths = self.pad_inputs(params)
                _, outp_lengths = field.pad(fake_input)
                self.assertEqual(outp_lengths, lengths)

    def test_pad_pads_right_places_and_uses_correct_index(self):
        for init_case, params in itertools.product(
                self.INIT_CASES, self.PARAMS):
            if not self.degenerate_case(init_case, params):
                field = AudioSeqField(**init_case)
                fake_input, lengths = self.pad_inputs(params)
                outp = field.pad(fake_input)
                if init_case["include_lengths"]:
                    outp, _ = outp
                for b in range(params["batch_size"]):
                    for s in range(lengths[b], params["max_len"]):
                        self.assertTrue(
                            outp[b, :, :, s].allclose(
                                torch.tensor(float(init_case["pad_index"]))))

    def test_numericalize_shape(self):
        for init_case, params in itertools.product(
                self.INIT_CASES, self.PARAMS):
            if not self.degenerate_case(init_case, params):
                field = AudioSeqField(**init_case)
                fake_input, lengths = self.numericalize_inputs(
                    init_case, params)
                outp = field.numericalize(fake_input)
                if init_case["include_lengths"]:
                    outp, _ = outp
                if init_case["batch_first"]:
                    expected_shape = (
                        params["batch_size"], 1,
                        params["nfeats"], params["max_len"])
                else:
                    expected_shape = (
                        params["max_len"], params["batch_size"],
                        1, params["nfeats"])
                self.assertEqual(expected_shape, outp.shape,
                                 init_case.__str__())

    def test_process_shape(self):
        # tests pad and numericalize integration
        for init_case, params in itertools.product(
                self.INIT_CASES, self.PARAMS):
            if not self.degenerate_case(init_case, params):
                field = AudioSeqField(**init_case)
                fake_input, lengths = self.pad_inputs(params)
                outp = field.process(fake_input)
                if init_case["include_lengths"]:
                    outp, _ = outp
                if init_case["batch_first"]:
                    expected_shape = (
                        params["batch_size"], 1,
                        params["nfeats"], params["max_len"])
                else:
                    expected_shape = (
                        params["max_len"], params["batch_size"],
                        1, params["nfeats"])
                self.assertEqual(expected_shape, outp.shape,
                                 init_case.__str__())

    def test_process_lengths(self):
        # tests pad and numericalize integration
        for init_case, params in itertools.product(
                self.INIT_CASES, self.PARAMS):
            if not self.degenerate_case(init_case, params):
                if init_case["include_lengths"]:
                    field = AudioSeqField(**init_case)
                    fake_input, lengths = self.pad_inputs(params)
                    lengths = torch.tensor(lengths, dtype=torch.int)
                    _, outp_lengths = field.process(fake_input)
                    self.assertTrue(outp_lengths.eq(lengths).all())


class TestAudioDataReader(unittest.TestCase):
    # this test touches the file system, so it could be considered an
    # integration test
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _AUDIO_DATA_DIRNAME = "test_audio_data"
    _AUDIO_DATA_DIR = os.path.join(_THIS_DIR, _AUDIO_DATA_DIRNAME)
    _AUDIO_DATA_FMT = "test_noise_{:d}.wav"
    _AUDIO_DATA_PATH_FMT = os.path.join(_AUDIO_DATA_DIR, _AUDIO_DATA_FMT)

    _AUDIO_LIST_DIR = "test_audio_filenames"
    # file to hold full paths to audio data
    _AUDIO_LIST_PATHS_FNAME = "test_files.txt"
    _AUDIO_LIST_PATHS_PATH = os.path.join(
        _AUDIO_LIST_DIR, _AUDIO_LIST_PATHS_FNAME)
    # file to hold audio paths relative to _AUDIO_DATA_DIR (i.e. file names)
    _AUDIO_LIST_FNAMES_FNAME = "test_fnames.txt"
    _AUDIO_LIST_FNAMES_PATH = os.path.join(
        _AUDIO_LIST_DIR, _AUDIO_LIST_FNAMES_FNAME)

    # it's ok if non-audio files co-exist with audio files in the data dir
    _JUNK_FILE = os.path.join(
        _AUDIO_DATA_DIR, "this_is_junk.txt")

    _N_EXAMPLES = 20
    _SAMPLE_RATE = 48000
    _N_CHANNELS = 2

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls._AUDIO_DATA_DIR):
            os.makedirs(cls._AUDIO_DATA_DIR)
        if not os.path.exists(cls._AUDIO_LIST_DIR):
            os.makedirs(cls._AUDIO_LIST_DIR)

        with open(cls._JUNK_FILE, "w") as f:
            f.write("this is some garbage\nShould have no impact.")

        with open(cls._AUDIO_LIST_PATHS_PATH, "w") as f_list_fnames, \
                open(cls._AUDIO_LIST_FNAMES_PATH, "w") as f_list_paths:
            lengths = torch.randint(int(.5e5), int(1.5e6), (cls._N_EXAMPLES,))
            for i in range(cls._N_EXAMPLES):
                # dividing gets the noise in [-1, 1]
                white_noise = torch.randn((cls._N_CHANNELS, lengths[i])) / 10
                f_path = cls._AUDIO_DATA_PATH_FMT.format(i)
                torchaudio.save(f_path, white_noise, cls._SAMPLE_RATE)
                f_name_short = cls._AUDIO_DATA_FMT.format(i)
                f_list_fnames.write(f_name_short + "\n")
                f_list_paths.write(f_path + "\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._AUDIO_DATA_DIR)
        shutil.rmtree(cls._AUDIO_LIST_DIR)

    def test_read_from_dir_and_data_file_containing_filenames(self):
        rdr = AudioDataReader(self._SAMPLE_RATE, window="hamming",
                              window_size=0.02, window_stride=0.01)
        i = 0  # initialize since there's a sanity check on i
        for i, aud in enumerate(rdr.read(
                self._AUDIO_LIST_FNAMES_PATH, "src", self._AUDIO_DATA_DIR)):
            self.assertEqual(aud["src"].shape[0], 481)
            self.assertEqual(aud["src_path"],
                             self._AUDIO_DATA_PATH_FMT.format(i))
        self.assertGreater(i, 0, "No audio data was read.")

    def test_read_from_dir_and_data_file_containing_paths(self):
        rdr = AudioDataReader(self._SAMPLE_RATE, window="hamming",
                              window_size=0.02, window_stride=0.01)
        i = 0  # initialize since there's a sanity check on i
        for i, aud in enumerate(rdr.read(
                self._AUDIO_LIST_PATHS_PATH, "src", self._AUDIO_DATA_DIR)):
            self.assertEqual(aud["src"].shape[0], 481)
            self.assertEqual(aud["src_path"],
                             self._AUDIO_DATA_FMT.format(i))
        self.assertGreater(i, 0, "No audio data was read.")
