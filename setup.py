#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='OpenNMT-py',
    description='A python implementation of OpenNMT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='1.0.0.rc1',
    packages=find_packages(),
    install_requires=[
        "six",
        "tqdm~=4.30.0",
        "torch>=1.2",
        "torchtext==0.4.0",
        "future",
        "configargparse",
        "tensorboard>=1.14",
        "flask",
        "pyonmttok",
    ],
    entry_points={
        "console_scripts": [
            "onmt_server=onmt.bin.server:main",
            "onmt_train=onmt.bin.train:main",
            "onmt_translate=onmt.bin.translate:main",
            "onmt_preprocess=onmt.bin.preprocess:main",
        ],
    }
)
