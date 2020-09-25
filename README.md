# OpenNMT-py: Open-Source Neural Machine Translation

[![Build Status](https://travis-ci.org/OpenNMT/OpenNMT-py.svg?branch=master)](https://travis-ci.org/OpenNMT/OpenNMT-py)
[![Run on FH](https://img.shields.io/badge/Run%20on-FloydHub-blue.svg)](https://floydhub.com/run?template=https://github.com/OpenNMT/OpenNMT-py)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://opennmt.net/OpenNMT-py/)
[![Gitter](https://badges.gitter.im/OpenNMT/OpenNMT-py.svg)](https://gitter.im/OpenNMT/OpenNMT-py?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Forum](https://img.shields.io/badge/discourse-online-green.svg)](https://forum.opennmt.net/)

This is the [PyTorch](https://github.com/pytorch/pytorch) version of the [OpenNMT](https://opennmt.net) project, an open-source (MIT) neural machine translation framework. It is designed to be research friendly to try out new ideas in translation, summary, morphology, and many other domains. Some companies have proven the code to be production ready.

We love contributions. Please consult the Issues page for any [Contributions Welcome](https://github.com/OpenNMT/OpenNMT-py/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22) tagged post. 

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>
Before raising an issue, make sure you read the requirements and the documentation examples.

Unless there is a bug, please use the [Forum](http://forum.opennmt.net) or [Gitter](https://gitter.im/OpenNMT/OpenNMT-py) to ask questions.

----

# Announcement - OpenNMT-py 2.0

**We're happy to announce the upcoming release v2.0 of OpenNMT-py.**

The major idea behind this release is the -- almost -- complete **makeover of the data loading pipeline**. A new 'dynamic' paradigm is introduced, allowing to apply on the fly transforms to the data.

This has a few advantages, amongst which:

- remove or drastically reduce the preprocessing required to train a model;
- increase the possibilities of data augmentation and manipulation through on-the fly transforms.

These transforms can be specific tokenization methods, filters, noising, or any custom transform users may want to implement. Custom transform implementation is quite straightforward thanks to the existing base class and example implementations.

You can check out how to use this new data loading pipeline in the updated [docs](https://opennmt.net/OpenNMT-py).

All the readily available transforms are described [here](https://opennmt.net/OpenNMT-py/FAQ.html#what-are-the-readily-available-on-the-fly-data-transforms).

### Performance

Given sufficient CPU resources according to GPU computing power, most of the transforms should not slow the training down. (Note: for now, one producer process per GPU is spawned -- meaning you would ideally need 2N CPU threads for N GPUs).

### Breaking changes

For now, the new data loading paradigm does not support Audio, Video and Image inputs.

A few features are also dropped, at least for now:

- audio, image and video inputs;
- source word features.

For any user that still need these features, the previous codebase will be retained as `legacy` in a separate branch. It will no longer receive extensive development from the core team but PRs may still be accepted.

Feel free to check it out and let us know what you think of the new paradigm!

----


Table of Contents
=================
  * [Full Documentation](http://opennmt.net/OpenNMT-py/)
  * [Usage](#usage)
  * [Features](#features)
  * [Quickstart](#quickstart)
  * [Run on FloydHub](#run-on-floydhub)
  * [Acknowledgements](#acknowledgements)
  * [Citation](#citation)

## Setup

OpenNMT-py requires:

- Python >= 3.5
- Pytorch == 1.6.0

Install `OpenNMT-py` from `pip`:
```bash
pip install OpenNMT-py
```

or from the sources:
```bash
git clone https://github.com/OpenNMT/OpenNMT-py.git
cd OpenNMT-py
python setup.py install
```

Note: If you have MemoryError in the install try to use `pip` with `--no-cache-dir`.

*(Optional)* some advanced features (e.g. working pretrained models or specific transforms) requires extra packages, you can install it with:

```bash
pip install -r requirements.opt.txt
```

## Features

- :warning: **New in OpenNMT-py 2.0**: [On the fly data processing]([here](https://opennmt.net/OpenNMT-py/FAQ.html#what-are-the-readily-available-on-the-fly-data-transforms).)

- [Seq2Seq models (encoder-decoder) with multiple RNN cells (lstm/gru) and attention (dotprod/mlp) types](http://opennmt.net/OpenNMT-py/options/train.html#model-encoder-decoder)
- [Transformer models](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-the-transformer-model)
- [Copy and Coverage Attention](http://opennmt.net/OpenNMT-py/options/train.html#model-attention)
- [Pretrained Embeddings](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-pretrained-embeddings-e-g-glove)
- [Source word features](http://opennmt.net/OpenNMT-py/options/train.html#model-embeddings)
- [TensorBoard logging](http://opennmt.net/OpenNMT-py/options/train.html#logging)
- [Multi-GPU training](http://opennmt.net/OpenNMT-py/FAQ.html##do-you-support-multi-gpu)
- [Data preprocessing](http://opennmt.net/OpenNMT-py/options/preprocess.html)
- [Inference (translation) with batching and beam search](http://opennmt.net/OpenNMT-py/options/translate.html)
- Inference time loss functions
- [Conv2Conv convolution model](https://arxiv.org/abs/1705.03122)
- SRU "RNNs faster than CNN" [paper](https://arxiv.org/abs/1709.02755)
- Mixed-precision training with [APEX](https://github.com/NVIDIA/apex), optimized on [Tensor Cores](https://developer.nvidia.com/tensor-cores)

## Quickstart

[Full Documentation](http://opennmt.net/OpenNMT-py/)

### Step 1: Prepare the data

To get started, we propose to download a toy English-German dataset for machine translation containing 10k tokenized sentences:

```bash
wget https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
tar xf toy-ende.tar.gz
cd toy-ende
```

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are used to evaluate the convergence of the training. It usually contains no more than 5k sentences.

```text
$ head -n 3 toy_ende/src-train.txt
It is not acceptable that , with the help of the national bureaucracies , Parliament &apos;s legislative prerogative should be made null and void by means of implementing provisions whose content , purpose and extent are not laid down in advance .
Federal Master Trainer and Senior Instructor of the Italian Federation of Aerobic Fitness , Group Fitness , Postural Gym , Stretching and Pilates; from 2004 , he has been collaborating with Antiche Terme as personal Trainer and Instructor of Stretching , Pilates and Postural Gym .
&quot; Two soldiers came up to me and told me that if I refuse to sleep with them , they will kill me . They beat me and ripped my clothes .
```

We need to build a **YAML configuration file** to specify the data that will be used:

```yaml
# toy_en_de.yaml

## Where the vocab(s) will be written
save_data: toy-ende/run/example
# Prevent overwriting existing files in the folder
overwrite: False

# Corpus opts:
data:
    corpus_1:
        path_src: toy-ende/src-train.txt
        path_tgt: toy-ende/tgt-train.txt
    valid:
        path_src: data/src-val.txt
        path_tgt: data/tgt-val.txt
...

```

From this configuration, we can build the vocab(s), that will be necessary to train the model:
```bash
onmt_build_vocab -config toy_en_de.yaml -n_sample 10000
```

**Notes**:
- `-n_sample` is required here -- it represents the number of lines sampled from each corpus to build the vocab.
- This configuration is the simplest possible, without any tokenization or other *transforms*. See [other example configurations](https://github.com/OpenNMT/OpenNMT-py/tree/master/config) for more complex pipelines.


### Step 2: Train the model

To train a model, we need to **add the following to the YAML configuration file**:
- the vocabulary path(s) that will be used;
- training specific parameters.

```yaml
# toy_en_de.yaml

...

# Vocabulary files that were just created
src_vocab: toy-ende/run/example.vocab.src
tgt_vocab: toy-ende/run/example.vocab.tgt

# Train on a single GPU
world_size: 1
gpu_ranks: [0]

# Where to save the checkpoints
save_model: toy-ende/run/model
save_checkpoint_steps: 500
train_steps: 1000
valid_steps: 500

```

Then you can simply run:

```bash
onmt_train -config toy_en_de.yaml
```

This configuration will run the default model, which consists of a 2-layer LSTM with 500 hidden units on both the encoder and decoder. It will run on a single GPU (`world_size 1` & `gpu_ranks [0]`).

Before the training process actually starts, the `*.vocab.pt` together with `*.transforms.pt` will be dumpped to `-save_data` with configurations specified in `-config` yaml file. We'll also generate transformed samples to simplify any potentially required visual inspection. The number of sample lines to dump per corpus is set with the `-n_sample` flag.

For more advanded models and parameters, see [other example configurations](https://github.com/OpenNMT/OpenNMT-py/tree/master/config) or the [FAQ](FAQ).

### Step 3: Translate

```bash
onmt_translate -model toy-ende/run/model_step_1000.pt -src toy-ende/src-test.txt -output toy-ende/pred_1000.txt -gpu 0 -verbose
```

Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into `toy-ende/pred_1000.txt`.

**Note**:

The predictions are going to be quite terrible, as the demo dataset is small. Try running on some larger datasets! For example you can download millions of parallel sentences for [translation](http://www.statmt.org/wmt16/translation-task.html) or [summarization](https://github.com/harvardnlp/sent-summary).

## Alternative: Run on FloydHub

[![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run?template=https://github.com/OpenNMT/OpenNMT-py)

Click this button to open a Workspace on [FloydHub](https://www.floydhub.com/?utm_medium=readme&utm_source=opennmt-py&utm_campaign=jul_2018) for training/testing your code.


## Pretrained embeddings (e.g. GloVe)

Please see the FAQ: [How to use GloVe pre-trained embeddings in OpenNMT-py](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-pretrained-embeddings-e-g-glove)

## Pretrained Models

The following pretrained models can be downloaded and used with translate.py.

http://opennmt.net/Models-py/

## Acknowledgements

OpenNMT-py is run as a collaborative open-source project.
The original code was written by [Adam Lerer](http://github.com/adamlerer) (NYC) to reproduce OpenNMT-Lua using Pytorch.

Major contributors are:
[Sasha Rush](https://github.com/srush) (Cambridge, MA)
[Vincent Nguyen](https://github.com/vince62s) (Ubiqus)
[Ben Peters](http://github.com/bpopeters) (Lisbon)
[Sebastian Gehrmann](https://github.com/sebastianGehrmann) (Harvard NLP)
[Yuntian Deng](https://github.com/da03) (Harvard NLP)
[Guillaume Klein](https://github.com/guillaumekln) (Systran)
[Paul Tardy](https://github.com/pltrdy) (Ubiqus / Lium)
[François Hernandez](https://github.com/francoishernandez) (Ubiqus)
[Linxiao Zeng](https://github.com/Zenglinxiao) (Ubiqus)
[Jianyu Zhan](http://github.com/jianyuzhan) (Shanghai)
[Dylan Flaute](http://github.com/flauted) (University of Dayton)
and more !

OpenNMT-py belongs to the OpenNMT project along with OpenNMT-Lua and OpenNMT-tf.

## Citation

[OpenNMT: Neural Machine Translation Toolkit](https://arxiv.org/pdf/1805.11462)

[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {Open{NMT}: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```