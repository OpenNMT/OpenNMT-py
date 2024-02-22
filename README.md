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

* You will need Pytorch v2 preferably v2.2 which fixes some `scaled_dot_product_attention` issues
* LLM support with converters for: Llama (+ Mistral), OpenLlama, Redpajama, MPT-7B, Falcon.
* Support for 8bit and 4bit quantization along with LoRA adapters, with or without checkpointing.
* You can finetune 7B and 13B models on a single RTX 24GB with 4-bit quantization.
* Inference can be forced in 4/8bit using the same layer quantization as in finetuning.
* Tensor parallelism when the model does not fit on one GPU's memory (both training and inference)
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

### Using docker

To facilitate setup and reproducibility, some docker images are made available via the Github Container Registry:
https://github.com/OpenNMT/OpenNMT-py/pkgs/container/opennmt-py

You can adapt the workflow and build your own image(s) depending on specific needs by using `build.sh` and `Dockerfile` in the `docker` directory of the repo.

```
docker pull ghcr.io/opennmt/opennmt-py:3.4.3-ubuntu22.04-cuda12.1
```

Example oneliner to run a container and open a bash shell within it
```
docker run --rm -it --runtime=nvidia ghcr.io/opennmt/opennmt-py:test-ubuntu22.04-cuda12.1
```
Note: you need to have the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (formerly nvidia-docker) installed to properly take advantage of the CUDA/GPU features.

Depending on your needs you can add various flags:
- `-p 5000:5000` to forward some exposed port from your container to your host;
- `-v /some/local/directory:/some/container/directory` to mount some local directory to some container directory;
- `--entrypoint some_command` to directly run some specific command as the container entry point (instead of the default bash shell);

### Installing locally

OpenNMT-py requires:

- Python >= 3.8
- PyTorch >= 2.0 <2.2

Install `OpenNMT-py` from `pip`:
```bash
pip install OpenNMT-py
```

or from the source:
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

### Manual installation of some dependencies

Apex is highly recommended to have fast performance (especially the legacy fusedadam optimizer and FusedRMSNorm)

```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --no-build-isolation --config-settings --build-option="--cpp_ext --cuda_ext --deprecated_fused_adam --xentropy --fast_multihead_attn" ./
cd ..
```

Flash attention:

As of Oct. 2023 flash attention 1 has been upstreamed to pytorch v2 but it is recommended to use flash attention 2 with v2.3.1 for sliding window attention support.

When using regular `position_encoding=True` or Rotary with `max_relative_positions=-1` OpenNMT-py will try to use an optimized dot-product path.

if you want to use [flash attention](https://github.com/Dao-AILab/flash-attention#installation-and-features) then you need to manually install it first:

```bash
pip install flash-attn --no-build-isolation
```

if flash attention 2 is not installed, then we will use `F.scaled_dot_product_attention` from pytorch 2.x

When using `max_relative_positions > 0` or Alibi `max_relative_positions=-2` OpenNMT-py will use its legacy code for matrix multiplications.

flash attention and `F.scaled_dot_product_attention` are a bit faster and saves some GPU memory.


AWQ:

If you want to run inference or quantize an AWQ model you will need AutoAWQ.

For [AutoAWQ](https://github.com/casper-hansen/AutoAWQ):
    pip install autoawq


## Documentation & FAQs

[Full HTML Documentation](https://opennmt.net/OpenNMT-py/quickstart.html)

[FAQs](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/FAQ.md)

## Acknowledgements

OpenNMT-py is run as a collaborative open-source project.
Project was incubated by Systran and Harvard NLP in 2016 in Lua and ported to Pytorch in 2017.

Current maintainers (since 2018):

[François Hernandez](https://github.com/francoishernandez)
[Vincent Nguyen](https://github.com/vince62s) (Seedfall)

## Citation

If you are using OpenNMT-py for academic work, please cite the initial [system demonstration paper](https://www.aclweb.org/anthology/P17-4012) published in ACL 2017:

```
@misc{klein2018opennmt,
      title={OpenNMT: Neural Machine Translation Toolkit}, 
      author={Guillaume Klein and Yoon Kim and Yuntian Deng and Vincent Nguyen and Jean Senellart and Alexander M. Rush},
      year={2018},
      eprint={1805.11462},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

