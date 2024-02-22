
**Notes on versioning**

## [Unreleased]

## [3.5.0](https://github.com/OpenNMT/OpenNMT-py/tree/3.5.0) (2024-02-22)

* Further improvements and fixes
* Suport for AWQ models
* Add n_best for topp/topk generation
* Support MoE (MIxtral) inference
* Extend HF models converter
* use flash_attn_with_kvcache for faster inference
* Add wikitext2 PPL computation
* Support for Phi-2 models

## [3.4.3](https://github.com/OpenNMT/OpenNMT-py/tree/3.4.3) (2023-11-2)

* Further improvements to beam search and decoding
* New indexing "in bucket" for faster inference cf #2496
* Code cleanup
* Fix int8 for CPU dynamic quantization (still slow...)

## [3.4.2](https://github.com/OpenNMT/OpenNMT-py/tree/3.4.2) (2023-10-20)

* torch 2.1 (scaled_dot_product improvements)
* Mistral 7B sliding window
* Speed-up inference
* flash attention 2 (with sliding window) >= v2.3.1
* use FusedRMSNorm from apex if available
* fixed attn_debug

## [3.4.1](https://github.com/OpenNMT/OpenNMT-py/tree/3.4.1) (2023-09-26)

* bug fixes
* torch 2.x requirement (flash attention requires it)
* zero-out the prompt loss in LM finetuning
* batching sorted on src then tgt instead of max len
* six dependancy

## [3.4.0](https://github.com/OpenNMT/OpenNMT-py/tree/3.4.0) (2023-09-06)

* bitsandbytes 4/8 bit quantization at inference
* MMLU-FR results and scoring
* flan-T5 support
* flash attention
* terminology transform
* tensor parallelism (inference, training)

## [3.3.0](https://github.com/OpenNMT/OpenNMT-py/tree/3.3.0) (2023-06-22)

* Switch to pytorch 2.0.1
* Eval LLM with MMLU benchmark
* Fix Falcon 40B conversion / finetuning / inference
* Plugin encoder/decoder thanks @kleag / @n2oblife
* Safetensors for model storage (beta)
* finetuning config templates for supported LLMs


## [3.2.0](https://github.com/OpenNMT/OpenNMT-py/tree/3.2.0) (2023-06-07)
* Skip init during model build (way faster building)
* Enable quantization of LoRA layers
* Enable 4bit quantization from bitsandbytes (NF4 / FP4)
* Enable "some" bnb.optim Optimizers for benchmarking purpose
* Refactor model state_dict loading to enable pseudo lazy loading with move on GPU as it loads
* Enable Gradient checkpointing for FFN, MHA, LoRA modules
* Make FFN bias optional (same as QKV): llama, mpt, redpajama, openllama converters changed accordingly.
  Convertv2_v3 set add_qkvbias=True, add_ffnbias=True.
  load_checkpoint: if w1_bias detected in checkpoint then add_ffnbias=True
* Add Multi Query attention
* Add Parallel Residual attention
* Add Falcon 7B converter

## [3.1.3](https://github.com/OpenNMT/OpenNMT-py/tree/3.1.3) (2023-05-24)
* Step-by-step Tuto for Vicuna replication thanks Lina
* MosaicML MPT7B converter and support (Alibi embeddings)
* Open Llama converter
* Switch GCLD3 to Fasttext thanks ArtanieTheOne
* fix coverage attention in beam decoding
* fix ct2 keys for "Llama / MPT7B based" OpenNMT-y models

## [3.1.2](https://github.com/OpenNMT/OpenNMT-py/tree/3.1.2) (2023-05-10)
* fixes: transforms (normalize, clean, inlinetags)
* Llama support (rotary embeddings, RMSNorm, Silu activation)
* 8bit loading for specific layers (along with LoRa for other layers)
* subword learner added to build_vocab

## [3.1.1](https://github.com/OpenNMT/OpenNMT-py/tree/3.1.1) (2023-03-30)
* fix major bug in 3.1.0 introduced with LoRa (3.1.0 not available)

## [3.1.0](https://github.com/OpenNMT/OpenNMT-py/tree/3.1.0) (2023-03-27)
* updated docs with Sphinx 6.4
* Restore source features to v3 (thanks @anderleich)
* add inline tags transform (thanks @panosk)
* add docify transform to allow doc-level training / inference
* fix NLLB training (decoder_start_token)
* New! LoRa adapters to finetune big models (egs: NLLB 3.3B)
* various bug fixes

## [3.0.4](https://github.com/OpenNMT/OpenNMT-py/tree/3.0.4) (2023-02-06)
* override_opts to override checkpoints opt when training from
* normalize transform based on (Sacre)Moses scripts
* uppercase transform for adhoc data augmentation
* suffix transform
* Fuzzy match transform
* WMT17 detailed example
* NLLB-200 (from Meta/FB) models support (after conversion)
* various bug fixes

## [3.0.3](https://github.com/OpenNMT/OpenNMT-py/tree/3.0.3) (2022-12-16)
* fix loss normalization when using accum or nb GPU > 1
* use native CrossEntropyLoss with Label Smoothing. reported loss/ppl impacted by LS
* fix long-time coverage loss bug thanks Sanghyuk-Choi
* fix detok at scoring / fix tokenization Subword_nmt + Sentencepiece
* various small bugs fixed

## [3.0.2](https://github.com/OpenNMT/OpenNMT-py/tree/3.0.2) (2022-12-07)
* pyonmttok.Vocab is now pickable. dataloader switched to spawn. (MacOS/Windows compatible)
* fix scoring with specific metrics (BLEU, TER)
* fix tensorboard logging
* fix dedup in batch iterator (only for TRAIN, was happening at inference also)
* New: Change: tgt_prefix renamed to tgt_file_prefix
* New: tgt_prefix / src_prefix used for "prefix" Transform (onmt/transforms/misc.py)
* New: process transforms of buckets in batches (vs per example) / faster

## [3.0.1](https://github.com/OpenNMT/OpenNMT-py/tree/3.0.1) (2022-11-23)

* fix dynamic scoring
* reinstate apex.amp level O1/O2 for benchmarking
* New: LM distillation for NMT training
* New: bucket_size ramp-up to avoid slow start
* fix special tokens order
* remove Library and add link to Yasmin's Tuto

## [3.0.0](https://github.com/OpenNMT/OpenNMT-py/tree/3.0.0) (2022-11-3)

* Removed completely torchtext. Use [Vocab object of pyonmttok](https://github.com/OpenNMT/Tokenizer/tree/master/bindings/python#vocabulary) instead
* Dataloading changed accordingly with the use of pytorch Dataloader (num_workers)
* queue_size / pool_factor no longer needed. bucket_size optimal value > 64K
* options renamed: rnn_size => hidden_size (enc/dec_rnn_size => enc/dec_hid_size)
* new tools/convertv2_v3.py to upgrade v2 models.pt
* inference with length_penalty=avg is now the default
* add_qkvbias (default false, but true for old model)

## [2.3.0](https://github.com/OpenNMT/OpenNMT-py/tree/2.3.0) (2022-09-14)

### New features
* BLEU/TER (& custom) scoring during training and validation (#2198)
* LM related tools (#2197)
* Allow encoder/decoder freezing (#2176)
* Dynamic data loading for inference (#2145)
* Sentence-level scores at inference (#2196)
* MBR and oracle reranking scoring tools (#2196)

### Fixes and improvements
* Updated beam exit condition (#2190)
* Improve scores reporting (#2191)
* Fix dropout scheduling (#2194)
* Better catch CUDA ooms when training (#2195)
* Fix source features support in inference and REST server (#2109)
* Make REST server more flexible with dictionaries (#2104)
* Fix target prefixing in LM decoding (#2099)

## [2.2.0](https://github.com/OpenNMT/OpenNMT-py/tree/2.2.0) (2021-09-14)

### New features
* Support source features (thanks @anderleich !)

### Fixes and improvements
* Adaptations to relax torch version
* Customizable transform statistics (#2059)
* Adapt release code for ctranslate2 2.0

## [2.1.2](https://github.com/OpenNMT/OpenNMT-py/tree/2.1.2) (2021-04-30)

### Fixes and improvements
*  Fix update_vocab for LM (#2056)

## [2.1.1](https://github.com/OpenNMT/OpenNMT-py/tree/2.1.1) (2021-04-30)

### Fixes and improvements
* Fix potential deadlock (b1a4615)
* Add more CT2 conversion checks (e4ab06c)

## [2.1.0](https://github.com/OpenNMT/OpenNMT-py/tree/2.1.0) (2021-04-16)

### New features
* Allow vocab update when training from a checkpoint (cec3cc8, 2f70dfc)

### Fixes and improvements
* Various transforms related bug fixes
* Fix beam warning and buffers reuse
* Handle invalid lines in vocab file gracefully

## [2.0.1](https://github.com/OpenNMT/OpenNMT-py/tree/2.0.1) (2021-01-27)

### Fixes and improvements
* Support embedding layer for larger vocabularies with GGNN (e8065b7)
* Reorganize some inference options (9fb5f30)

## [2.0.0](https://github.com/OpenNMT/OpenNMT-py/tree/2.0.0) (2021-01-20)

First official release for OpenNMT-py major upgdate to 2.0!

### New features
* Language Model (GPT-2 style) training and inference
* Nucleus (top-p) sampling decoding

### Fixes and improvements
* Fix some BART default values

## [2.0.0rc2](https://github.com/OpenNMT/OpenNMT-py/tree/2.0.0rc2) (2020-11-10)

### Fixes and improvements
* Parallelize onmt_build_vocab (422d824)
* Some fixes to the on-the-fly transforms
* Some CTranslate2 related updates
* Some fixes to the docs

## [2.0.0rc1](https://github.com/OpenNMT/OpenNMT-py/tree/2.0.0rc1) (2020-09-25)

This is the first release candidate for OpenNMT-py major upgdate to 2.0.0!

The major idea behind this release is the -- almost -- complete **makeover of the data loading pipeline** . A new 'dynamic' paradigm is introduced, allowing to apply on the fly transforms to the data.

This has a few advantages, amongst which:

* remove or drastically reduce the preprocessing required to train a model;
* increase and simplify the possibilities of data augmentation and manipulation through on-the fly transforms.

These transforms can be specific **tokenization** methods, **filters**, **noising**, or **any custom transform** users may want to implement. Custom transform implementation is quite straightforward thanks to the existing base class and example implementations.

You can check out how to use this new data loading pipeline in the updated [docs and examples](https://opennmt.net/OpenNMT-py).

All the **readily available transforms** are described [here](https://opennmt.net/OpenNMT-py/FAQ.html#what-are-the-readily-available-on-the-fly-data-transforms).

### Performance

Given sufficient CPU resources according to GPU computing power, most of the transforms should not slow the training down. (Note: for now, one producer process per GPU is spawned -- meaning you would ideally need 2N CPU threads for N GPUs).

### Breaking changes

A few features are dropped, at least for now:

* audio, image and video inputs;
* source word features.

Some very old checkpoints with previous fields and vocab structure are also incompatible with this new version.

For any user that still need some of these features, the previous codebase will be retained as [`legacy` in a separate branch](https://github.com/OpenNMT/OpenNMT-py/tree/legacy). It will no longer receive extensive development from the core team but PRs may still be accepted.


-----

## [1.2.0](https://github.com/OpenNMT/OpenNMT-py/tree/1.2.0) (2020-08-17)
### Fixes and improvements
* Support pytorch 1.6 (e813f4d, eaaae6a)
* Support official torch 1.6 AMP for mixed precision training (2ac1ed0)
* Flag to override batch_size_multiple in FP16 mode, useful in some memory constrained setups (23e5018)
* Pass a dict and allow custom options in preprocess/postprocess functions of REST server (41f0c02, 8ec54d2)
* Allow different tokenization for source and target in REST server (bb2d045, 4659170)
* Various bug fixes

### New features
* Gated Graph Sequence Neural Networks encoder (11e8d0), thanks @SteveKommrusch
* Decoding with a target prefix (95aeefb, 0e143ff, 91ab592), thanks @Zenglinxiao

## [1.1.1](https://github.com/OpenNMT/OpenNMT-py/tree/1.1.1) (2020-03-20)
### Fixes and improvements
* Fix backcompatibility when no 'corpus_id' field (c313c28)

## [1.1.0](https://github.com/OpenNMT/OpenNMT-py/tree/1.1.0) (2020-03-19)
### New features
* Support CTranslate2 models in REST server (91d5d57)
* Extend support for custom preprocessing/postprocessing function in REST server by using return dictionaries (d14613d, 9619ac3, 92a7ba5)
* Experimental: BART-like source noising (5940dcf)

### Fixes and improvements
* Add options to CTranslate2 release (e442f3f)
* Fix dataset shard order (458fc48)
* Rotate only the server logs, not training (189583a)
* Fix alignment error with empty prediction (91287eb)

## [1.0.2](https://github.com/OpenNMT/OpenNMT-py/tree/1.0.2) (2020-03-05)
### Fixes and improvements
* Enable CTranslate2 conversion of Transformers with relative position (db11135)
* Adapt `-replace_unk` to use with learned alignments if they exist (7625b53)

## [1.0.1](https://github.com/OpenNMT/OpenNMT-py/tree/1.0.1) (2020-02-17)
### Fixes and improvements
* Ctranslate2 conversion handled in release script (1b50e0c)
* Use `attention_dropout` properly in MHA (f5c9cd4)
* Update apex FP16_Optimizer path (d3e2268)
* Some REST server optimizations
* Fix and add some docs

## [1.0.0](https://github.com/OpenNMT/OpenNMT-py/tree/1.0.0) (2019-10-01)
### New features
* Implementation of "Jointly Learning to Align & Translate with Transformer" (@Zenglinxiao)

### Fixes and improvements
* Add nbest support to REST server (@Zenglinxiao)
* Merge greedy and beam search codepaths (@Zenglinxiao)
* Fix "block ngram repeats" (@KaijuML, @pltrdy)
* Small fixes, some more docs

## [1.0.0.rc2](https://github.com/OpenNMT/OpenNMT-py/tree/1.0.0.rc1) (2019-10-01)
* Fix Apex / FP16 training (Apex new API is buggy)
* Multithread preprocessing way faster (Thanks @francoishernandez)
* Pip Installation v1.0.0.rc1 (thanks @pltrdy)

## [0.9.2](https://github.com/OpenNMT/OpenNMT-py/tree/0.9.2) (2019-09-04)
* Switch to Pytorch 1.2
* Pre/post processing on the translation server
* option to remove the FFN layer in AAN + AAN optimization (faster)
* Coverage loss (per Abisee paper 2017) implementation
* Video Captioning task: Thanks Dylan Flaute!
* Token batch at inference
* Small fixes and add-ons


## [0.9.1](https://github.com/OpenNMT/OpenNMT-py/tree/0.9.1) (2019-06-13)
* New mechanism for MultiGPU training "1 batch producer / multi batch consumers"
  resulting in big memory saving when handling huge datasets
* New APEX AMP (mixed precision) API
* Option to overwrite shards when preprocessing
* Small fixes and add-ons

## [0.9.0](https://github.com/OpenNMT/OpenNMT-py/tree/0.9.0) (2019-05-16)
* Faster vocab building when processing shards (no reloading)
* New dataweighting feature
* New dropout scheduler.
* Small fixes and add-ons

## [0.8.2](https://github.com/OpenNMT/OpenNMT-py/tree/0.8.2) (2019-02-16)
* Update documentation and Library example
* Revamp args
* Bug fixes, save moving average in FP32
* Allow FP32 inference for FP16 models

## [0.8.1](https://github.com/OpenNMT/OpenNMT-py/tree/0.8.1) (2019-02-12)
* Update documentation
* Random sampling scores fixes
* Bug fixes

## [0.8.0](https://github.com/OpenNMT/OpenNMT-py/tree/0.8.0) (2019-02-09)
* Many fixes and code cleaning thanks @flauted, @guillaumekln
* Datasets code refactor (thanks @flauted) you need to r-preeprocess datasets

### New features
* FP16 Support: Experimental, using Apex, Checkpoints may break in future version.
* Continuous exponential moving average (thanks @francoishernandez, and Marian)
* Relative positions encoding (thanks @francoishernanndez, and Google T2T)
* Deprecate the old beam search, fast batched beam search supports all options


## [0.7.2](https://github.com/OpenNMT/OpenNMT-py/tree/0.7.2) (2019-01-31)
* Many fixes and code cleaning thanks @bpopeters, @flauted, @guillaumekln

### New features
* Multilevel fields for better handling of text featuer embeddinggs. 


## [0.7.1](https://github.com/OpenNMT/OpenNMT-py/tree/0.7.1) (2019-01-24)
* Many fixes and code refactoring thanks @bpopeters, @flauted, @guillaumekln

### New features
* Random sampling thanks @daphnei
* Enable sharding for huge files at translation

## [0.7.0](https://github.com/OpenNMT/OpenNMT-py/tree/0.7.0) (2019-01-02)
* Many fixes and code refactoring thanks @benopeters
* Migrated to Pytorch 1.0

## [0.6.0](https://github.com/OpenNMT/OpenNMT-py/tree/0.6.0) (2018-11-28)
* Many fixes and code improvements
* New: Ability to load a yml config file. See examples in config folder.

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


