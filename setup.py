#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="OpenNMT-py",
    description="A python implementation of OpenNMT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="3.5.0",
    packages=find_packages(),
    project_urls={
        "Documentation": "http://opennmt.net/OpenNMT-py/",
        "Forum": "http://forum.opennmt.net/",
        "Gitter": "https://gitter.im/OpenNMT/OpenNMT-py",
        "Source": "https://github.com/OpenNMT/OpenNMT-py/",
    },
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.1,<2.3",
        "configargparse",
        "ctranslate2>=3.24,<5",
        "tensorboard>=2.3",
        "flask",
        "waitress",
        "pyonmttok>=1.35,<2",
        "pyyaml",
        "sacrebleu",
        "rapidfuzz",
        "pyahocorasick",
        "fasttext-wheel",
        "spacy",
        "six",
    ],
    entry_points={
        "console_scripts": [
            "onmt_server=onmt.bin.server:main",
            "onmt_train=onmt.bin.train:main",
            "onmt_translate=onmt.bin.translate:main",
            "onmt_release_model=onmt.bin.release_model:main",
            "onmt_average_models=onmt.bin.average_models:main",
            "onmt_build_vocab=onmt.bin.build_vocab:main",
        ],
    },
)
