#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name='OpenNMT-py',
    description='A python implementation of OpenNMT',
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
