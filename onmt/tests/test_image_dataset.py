import unittest
from onmt.inputters.image_dataset import ImageDataReader

import os
import shutil

import cv2
import numpy as np
import torch


class TestImageDataReader(unittest.TestCase):
    # this test touches the file system, so it could be considered an
    # integration test
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _IMG_DATA_DIRNAME = "test_image_data"
    _IMG_DATA_DIR = os.path.join(_THIS_DIR, _IMG_DATA_DIRNAME)
    _IMG_DATA_FMT = "test_img_{:d}.png"
    _IMG_DATA_PATH_FMT = os.path.join(_IMG_DATA_DIR, _IMG_DATA_FMT)

    _IMG_LIST_DIR = "test_image_filenames"
    # file to hold full paths to image data
    _IMG_LIST_PATHS_FNAME = "test_files.txt"
    _IMG_LIST_PATHS_PATH = os.path.join(
        _IMG_LIST_DIR, _IMG_LIST_PATHS_FNAME)
    # file to hold image paths relative to _IMG_DATA_DIR (i.e. file names)
    _IMG_LIST_FNAMES_FNAME = "test_fnames.txt"
    _IMG_LIST_FNAMES_PATH = os.path.join(
        _IMG_LIST_DIR, _IMG_LIST_FNAMES_FNAME)

    # it's ok if non-image files co-exist with image files in the data dir
    _JUNK_FILE = os.path.join(
        _IMG_DATA_DIR, "this_is_junk.txt")

    _N_EXAMPLES = 20
    _N_CHANNELS = 3

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls._IMG_DATA_DIR):
            os.makedirs(cls._IMG_DATA_DIR)
        if not os.path.exists(cls._IMG_LIST_DIR):
            os.makedirs(cls._IMG_LIST_DIR)

        with open(cls._JUNK_FILE, "w") as f:
            f.write("this is some garbage\nShould have no impact.")

        with open(cls._IMG_LIST_PATHS_PATH, "w") as f_list_fnames, \
                open(cls._IMG_LIST_FNAMES_PATH, "w") as f_list_paths:
            cls.n_rows = torch.randint(30, 314, (cls._N_EXAMPLES,))
            cls.n_cols = torch.randint(30, 314, (cls._N_EXAMPLES,))
            for i in range(cls._N_EXAMPLES):
                img = np.random.randint(
                    0, 255, (cls.n_rows[i], cls.n_cols[i], cls._N_CHANNELS))
                f_path = cls._IMG_DATA_PATH_FMT.format(i)
                cv2.imwrite(f_path, img)
                f_name_short = cls._IMG_DATA_FMT.format(i)
                f_list_fnames.write(f_name_short + "\n")
                f_list_paths.write(f_path + "\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._IMG_DATA_DIR)
        shutil.rmtree(cls._IMG_LIST_DIR)

    def test_read_from_dir_and_data_file_containing_filenames(self):
        rdr = ImageDataReader(channel_size=self._N_CHANNELS)
        i = 0  # initialize since there's a sanity check on i
        for i, img in enumerate(rdr.read(
                self._IMG_LIST_FNAMES_PATH, "src", self._IMG_DATA_DIR)):
            self.assertEqual(
                img["src"].shape,
                (self._N_CHANNELS, self.n_rows[i], self.n_cols[i]))
            self.assertEqual(img["src_path"],
                             self._IMG_DATA_PATH_FMT.format(i))
        self.assertGreater(i, 0, "No image data was read.")

    def test_read_from_dir_and_data_file_containing_paths(self):
        rdr = ImageDataReader(channel_size=self._N_CHANNELS)
        i = 0  # initialize since there's a sanity check on i
        for i, img in enumerate(rdr.read(
                self._IMG_LIST_PATHS_PATH, "src", self._IMG_DATA_DIR)):
            self.assertEqual(
                img["src"].shape,
                (self._N_CHANNELS, self.n_rows[i], self.n_cols[i]))
            self.assertEqual(img["src_path"],
                             self._IMG_DATA_FMT.format(i))
        self.assertGreater(i, 0, "No image data was read.")


class TestImageDataReader1Channel(TestImageDataReader):
    _N_CHANNELS = 1
