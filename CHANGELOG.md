
**Notes on versioning**


## [Unreleased]

### New features

### Fixes and improvements


## [0.2.1](https://github.com/OpenNMT/OpenNMT-py/tree/v0.2.1) (2018-08-31)

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


