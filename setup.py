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
    version='2.0.0rc2',
    packages=find_packages(),
    project_urls={
        "Documentation": "http://opennmt.net/OpenNMT-py/",
        "Forum": "http://forum.opennmt.net/",
        "Gitter": "https://gitter.im/OpenNMT/OpenNMT-py",
        "Source": "https://github.com/OpenNMT/OpenNMT-py/"
    },
    install_requires=[
        "numpy==1.19.3",
        "six==1.15.0",
        "tqdm==4.51.0",
        "torch==1.6.0",
        "torchtext==0.5.0",
        "future==0.18.2",
        "configargparse==1.2.3",
        "tensorboard==2.3.0",
        "flask==1.1.2",
        "waitress==1.4.4",
        "pyonmttok>=1.23,<2;platform_system=='Linux' or platform_system=='Darwin'",
        "pyyaml==5.3.1",
    ],
    entry_points={
        "console_scripts": [
            "onmt_server=onmt.bin.server:main",
            "onmt_train=onmt.bin.train:main",
            "onmt_translate=onmt.bin.translate:main",
            "onmt_release_model=onmt.bin.release_model:main",
            "onmt_average_models=onmt.bin.average_models:main",
            "onmt_build_vocab=onmt.bin.build_vocab:main"
        ],
    }
)
