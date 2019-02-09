<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

preprocess.py
# Options: preprocess.py:
preprocess.py Args that start with '--' (eg. --md) can also be set in a config
file (specified via -config). The config file uses YAML syntax and must
represent a YAML 'mapping' (for details, see
http://learn.getgrav.org/advanced/yaml). If an arg is specified in more than
one place, then commandline values override config file values which override
defaults.

### **optional arguments**:
* **-config []** 
config file path

* **-save_config []** 
config file save path

### **Data**:
* **-data_type [text]** 
Type of the source input. Options are [text|img|audio].

* **-train_src []** 
Path to the training source data

* **-train_tgt []** 
Path to the training target data

* **-valid_src []** 
Path to the validation source data

* **-valid_tgt []** 
Path to the validation target data

* **-src_dir []** 
Source directory for image or audio files.

* **-save_data []** 
Output file for the prepared data

* **-max_shard_size []** 
Deprecated use shard_size instead

* **-shard_size [1000000]** 
Divide src_corpus and tgt_corpus into smaller multiple src_copus and tgt corpus
files, then build shards, each shard will have opt.shard_size samples except
last shard. shard_size=0 means no segmentation shard_size>0 means segment
dataset into multiple shards, each shard has shard_size samples

### **Vocab**:
* **-src_vocab []** 
Path to an existing source vocabulary. Format: one word per line.

* **-tgt_vocab []** 
Path to an existing target vocabulary. Format: one word per line.

* **-features_vocabs_prefix []** 
Path prefix to existing features vocabularies

* **-src_vocab_size [50000]** 
Size of the source vocabulary

* **-tgt_vocab_size [50000]** 
Size of the target vocabulary

* **-vocab_size_multiple [1]** 
Make the vocabulary size a multiple of this value

* **-src_words_min_frequency []** 

* **-tgt_words_min_frequency []** 

* **-dynamic_dict []** 
Create dynamic dictionaries

* **-share_vocab []** 
Share source and target vocabulary

### **Pruning**:
* **-src_seq_length [50]** 
Maximum source sequence length

* **-src_seq_length_trunc []** 
Truncate source sequence length.

* **-tgt_seq_length [50]** 
Maximum target sequence length to keep.

* **-tgt_seq_length_trunc []** 
Truncate target sequence length.

* **-lower []** 
lowercase data

* **-filter_valid []** 
Filter validation data by src and/or tgt length

### **Random**:
* **-shuffle []** 
Shuffle data

* **-seed [3435]** 
Random seed

### **Logging**:
* **-report_every [100000]** 
Report status every this many sentences

* **-log_file []** 
Output logs to a file under this path.

* **-log_file_level [0]** 

### **Speech**:
* **-sample_rate [16000]** 
Sample rate.

* **-window_size [0.02]** 
Window size for spectrogram in seconds.

* **-window_stride [0.01]** 
Window stride for spectrogram in seconds.

* **-window [hamming]** 
Window type for spectrogram generation.

* **-image_channel_size [3]** 
Using grayscale image can training model faster and smaller
