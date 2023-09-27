# OpenNMT-py: Open-Source Neural Machine Translation and (Large) Language Models

[![Build Status](https://github.com/OpenNMT/OpenNMT-py/workflows/Lint%20&%20Tests/badge.svg)](https://github.com/OpenNMT/OpenNMT-py/actions)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://opennmt.net/OpenNMT-py/)
[![Gitter](https://badges.gitter.im/OpenNMT/OpenNMT-py.svg)](https://gitter.im/OpenNMT/OpenNMT-py?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Forum](https://img.shields.io/discourse/status?server=https%3A%2F%2Fforum.opennmt.net%2F)](https://forum.opennmt.net/)

OpenNMT-py is the [PyTorch](https://github.com/pytorch/pytorch) version of the [OpenNMT](https://opennmt.net) project, an open-source (MIT) neural machine translation (and beyond!) framework. It is designed to be research friendly to try out new ideas in translation, language modeling, summarization, and many other NLP tasks. Some companies have proven the code to be production ready.

We love contributions! Please look at issues marked with the [contributions welcome](https://github.com/OpenNMT/OpenNMT-py/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22) tag.

Before raising an issue, make sure you read the requirements and the [Full Documentation](https://opennmt.net/OpenNMT-py/) examples.

Unless there is a bug, please use the [Forum](https://forum.opennmt.net) or [Gitter](https://gitter.im/OpenNMT/OpenNMT-py) to ask questions.

----
## For beginners:

There is a step-by-step and explained tuto (Thanks to Yasmin Moslem): [Tutorial](https://github.com/ymoslem/OpenNMT-Tutorial)

Please try to read and/or follow before raising newbies issues.

Otherwise you can just have a look at the [Quickstart](https://opennmt.net/OpenNMT-py/quickstart.html) steps

----
## New:

* Special note on Pytorch v2: up to v2.0.1 dynamic shapes are not handled properly, hence torch.compile() will not work with OpenNMT-py. We have tested nightly (in May) and it works with a small gain. Next version will be 2.1
* LLM support with converters for: Llama, OpenLlama, Redpajama, MPT-7B, Falcon.
* Support for 8bit and 4bit quantization along with LoRA adapters, with or without checkpointing.
* You can finetune 7B and 13B models on a single RTX 24GB with 4-bit quantization.
* Inference can be forced in 4/8bit using the same layer quantization as in finetuning.
* Once your model is finetuned you can run inference either with OpenNMT-py or faster with CTranslate2.
* MMLU evaluation script, see results [here](https://github.com/OpenNMT/OpenNMT-py/blob/master/eval_llm/MMLU/readme.md)

For all usecases including NMT, you can now use Multiquery instead of Multihead attention (faster at training and inference) and remove biases from all Linear (QKV as well as FeedForward modules).


If you used previous versions of OpenNMT-py, you can check the [Changelog](https://github.com/OpenNMT/OpenNMT-py/blob/master/CHANGELOG.md) or the [Breaking Changes](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/changes.md)

----

## Tutorials:

* How to replicate Vicuna with a 7B or 13B llama (or Open llama, MPT-7B, Redpajama)  Language Model: [Tuto Vicuna](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/examples/replicate_vicuna/ReplicateVicuna.md)
* How to finetune NLLB-200 with your dataset: [Tuto Finetune NLLB-200](https://forum.opennmt.net/t/finetuning-and-curating-nllb-200-with-opennmt-py/5238)
* How to create a simple OpenNMT-py REST Server: [Tuto REST](https://forum.opennmt.net/t/simple-opennmt-py-rest-server/1392)
* How to create a simple Web Interface: [Tuto Streamlit](https://forum.opennmt.net/t/simple-web-interface/4527)
* Replicate the WMT17 en-de experiment: [WMT17 ENDE](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/examples/wmt17/Translation.md)

----

## Setup

OpenNMT-py requires:

- Python >= 3.8
- PyTorch >= 2.0 <2.1

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

## Documentation & FAQs

[Full HTML Documentation](https://opennmt.net/OpenNMT-py/quickstart.html)

[FAQs](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/FAQ.md)

## Acknowledgements

OpenNMT-py is run as a collaborative open-source project.
Project was incubated by Systran and Harvard NLP in 2016 in Lua and ported to Pytorch in 2017.

Current maintainers (since 2018):

[François Hernandez](https://github.com/francoishernandez) and Ubiqus Team.
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

