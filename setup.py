#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name='OpenNMT-py',
    description='A python implementation of OpenNMT',
    version='0.9.2',
    packages=find_packages(),
    install_requires=[
        "six",
        "tqdm~=4.30.0",
        "torch>=1.1",
        "torchtext==0.4.0",
        "future",
        "configargparse",
        "tensorboard>=1.14",
        "flask",
        "pyonmttok",
    ],
    entry_points={
        "console_scripts": [
            "onmt_server=server:main",
            "onmt_train=train:main",
            "onmt_translate=translate:main",
            "onmt_preprocess=preprocess:main",
        ],
    }
)
