
**Notes on versioning**


## [Unreleased]

### New features

### Fixes and improvements

## [0.5.0](https://github.com/OpenNMT/OpenNMT-py/tree/0.5.0) (2018-10-24)
* Fixed advance n_best beam in translate_batch_fast
* Fixed remove valid set vocab from total vocab
* New: Ability to reset optimizer when using train_from
* New: create_vocabulary tool + fix when loading existing vocab.

## [0.4.1](https://github.com/OpenNMT/OpenNMT-py/tree/0.4.1) (2018-10-11)
* Fixed preprocessing files names, cleaning intermediary files.

## [0.4.0](https://github.com/OpenNMT/OpenNMT-py/tree/0.4.0) (2018-10-08)
* Fixed Speech2Text training (thanks Yuntian)

* Removed -max_shard_size, replaced by -shard_size = number of examples in a shard.
  Default value = 1M which works fine in most Text dataset cases. (will avoid Ram OOM in most cases)


## [0.3.0](https://github.com/OpenNMT/OpenNMT-py/tree/0.3.0) (2018-09-27)
* Now requires Pytorch 0.4.1

* Multi-node Multi-GPU with Torch Distributed

  New options are:
  -master_ip: ip address of the master node
  -master_port: port number of th emaster node
  -world_size = total number of processes to be run (total GPUs accross all nodes)
  -gpu_ranks = list of indices of processes accross all nodes

* gpuid is deprecated
See examples in https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/FAQ.md

* Fixes to img2text now working

* New sharding based on number of examples

* Fixes to avoid 0.4.1 deprecated functions.


## [0.2.1](https://github.com/OpenNMT/OpenNMT-py/tree/0.2.1) (2018-08-31)

### Fixes and improvements

* First compatibility steps with Pytorch 0.4.1 (non breaking)
* Fix TranslationServer (when various request try to load the same model at the same time)
* Fix StopIteration error (python 3.7)

### New features
* Ensemble at inference (thanks @Waino)

## [0.2](https://github.com/OpenNMT/OpenNMT-py/tree/v0.2) (2018-08-28)

### improvements

* Compatibility fixes with Pytorch 0.4 / Torchtext 0.3
* Multi-GPU based on Torch Distributed
* Average Attention Network (AAN) for the Transformer (thanks @francoishernandez )
* New fast beam search (see -fast in translate.py) (thanks @guillaumekln)
* Sparse attention / sparsemax (thanks to @bpopeters)
* Refactoring of many parts of the code base:
 - change from -epoch to -train_steps -valid_steps (see opts.py)
 - reorg of the logic train => train_multi / train_single => trainer
* Many fixes / improvements in the translationserver (thanks @pltrdy @francoishernandez)
* fix BPTT

## [0.1](https://github.com/OpenNMT/OpenNMT-py/tree/v0.1) (2018-06-08)

### First and Last Release using Pytorch 0.3.x


