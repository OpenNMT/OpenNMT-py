# OpenNMT-py: Open-Source Neural Machine Translation

[![Build Status](https://github.com/OpenNMT/OpenNMT-py/workflows/Lint%20&%20Tests/badge.svg)](https://github.com/OpenNMT/OpenNMT-py/actions)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://opennmt.net/OpenNMT-py/)
[![Gitter](https://badges.gitter.im/OpenNMT/OpenNMT-py.svg)](https://gitter.im/OpenNMT/OpenNMT-py?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Forum](https://img.shields.io/discourse/status?server=https%3A%2F%2Fforum.opennmt.net%2F)](https://forum.opennmt.net/)

OpenNMT-py is the [PyTorch](https://github.com/pytorch/pytorch) version of the [OpenNMT](https://opennmt.net) project, an open-source (MIT) neural machine translation (and beyond!) framework. It is designed to be research friendly to try out new ideas in translation, language modeling, summary, morphology, and many other NLP tasks. Some companies have proven the code to be production ready.

We love contributions! Please look at issues marked with the [contributions welcome](https://github.com/OpenNMT/OpenNMT-py/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22) tag.

Before raising an issue, make sure you read the requirements and the [Full Documentation](https://opennmt.net/OpenNMT-py/) examples.

Unless there is a bug, please use the [forum](https://forum.opennmt.net) or [Gitter](https://gitter.im/OpenNMT/OpenNMT-py) to ask questions.

----
For beginners:

There is a new step-by-step and explained tuto (Thanks to Yasmin Moslem) here:
Please try to read and/or follow before raising newbie issues [Tutorial](https://github.com/ymoslem/OpenNMT-Tutorial)

Otherwise you can just have a look at the Quickstart steps.

----

If you used previous versions of OpenNMT-py, you can check the [Changelog](https://github.com/OpenNMT/OpenNMT-py/blob/master/CHANGELOG.md) or the [Breaking Changes](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/changes.md)

----

Tutorials:
* How to finetune NLLB-200 with your dataset: [Tuto Finetune NLLB-200](https://forum.opennmt.net/t/finetuning-and-curating-nllb-200-with-opennmt-py/5238)
* How to create a simple OpenNMT-py REST Server: [Tuto REST](https://forum.opennmt.net/t/simple-opennmt-py-rest-server/1392)
* How to create a simple Web Interface: [Tuto Streamlit](https://forum.opennmt.net/t/simple-web-interface/4527)

* Replicate the WMT17 en-de experiment: [WMT17 ENDE](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/examples/wmt17/Translation.md)

----
Table of Contents
=================
  * [Setup](#setup)
  * [Features](#features)
  * [Quickstart](#quickstart)
  * [Pretrained embeddings](#pretrained-embeddings-eg-glove)
  * [Pretrained models](#pretrained-models)
  * [Acknowledgements](#acknowledgements)
  * [Citation](#citation)

## Setup

OpenNMT-py requires:

- Python >= 3.8
- PyTorch >= 1.13 <2

Install `OpenNMT-py` from `pip`:
```bash
pip install OpenNMT-py
```

or from the sources:
```bash
git clone https://github.com/OpenNMT/OpenNMT-py.git
cd OpenNMT-py
pip install -e .
```

Note: if you encounter a `MemoryError` during installation, try to use `pip` with `--no-cache-dir`.

*(Optional)* Some advanced features (e.g. working pretrained models or specific transforms) require extra packages, you can install them with:

```bash
pip install -r requirements.opt.txt
```

## Features

- [End-to-end training with on-the-fly data processing]([here](https://opennmt.net/OpenNMT-py/FAQ.html#what-are-the-readily-available-on-the-fly-data-transforms).)

- [Transformer models](https://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-the-transformer-model)
- [Encoder-decoder models with multiple RNN cells (LSTM, GRU) and attention types (Luong, Bahdanau)](https://opennmt.net/OpenNMT-py/options/train.html#model-encoder-decoder)
- SRU "RNNs faster than CNN" [paper](https://arxiv.org/abs/1709.02755)
- [Conv2Conv convolution model](https://arxiv.org/abs/1705.03122)
- [Copy and Coverage Attention](https://opennmt.net/OpenNMT-py/options/train.html#model-attention)
- [Pretrained Embeddings](https://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-pretrained-embeddings-e-g-glove)
- [Source word features](https://opennmt.net/OpenNMT-py/options/train.html#model-embeddings)
- [TensorBoard logging](https://opennmt.net/OpenNMT-py/options/train.html#logging)
- Mixed-precision training with [APEX](https://github.com/NVIDIA/apex), optimized on [Tensor Cores](https://developer.nvidia.com/tensor-cores)
- [Multi-GPU training](https://opennmt.net/OpenNMT-py/FAQ.html##do-you-support-multi-gpu)

- [Inference (translation) with batching and beam search](https://opennmt.net/OpenNMT-py/options/translate.html)

- Model export to [CTranslate2](https://github.com/OpenNMT/CTranslate2), a fast and efficient inference engine

## Documentation

[Full Documentation](https://opennmt.net/OpenNMT-py/quickstart.html)

## Acknowledgements

OpenNMT-py is run as a collaborative open-source project.
Project was incubated by Systran and Harvard NLP in 2016 in Lua and ported to Pytorch in 2017.

Current maintainers:
Ubiqus Team: [François Hernandez](https://github.com/francoishernandez) and Team.
[Vincent Nguyen](https://github.com/vince62s) (Seedfall)

## Citation

If you are using OpenNMT-py for academic work, please cite the initial [system demonstration paper](https://www.aclweb.org/anthology/P17-4012) published in ACL 2017:

```
@inproceedings{klein-etal-2017-opennmt,
    title = "{O}pen{NMT}: Open-Source Toolkit for Neural Machine Translation",
    author = "Klein, Guillaume  and
      Kim, Yoon  and
      Deng, Yuntian  and
      Senellart, Jean  and
      Rush, Alexander",
    booktitle = "Proceedings of {ACL} 2017, System Demonstrations",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P17-4012",
    pages = "67--72",
}
```
