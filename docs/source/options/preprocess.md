<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

preprocess.py
# Options: preprocess.py:
preprocess.py

### **Data**:
* **-data_type [text]** 
Type of the source input. Options are [text|img].

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

### **Vocab**:
* **-src_vocab []** 
Path to an existing source vocabulary

* **-tgt_vocab []** 
Path to an existing target vocabulary

* **-features_vocabs_prefix []** 
Path prefix to existing features vocabularies

* **-src_vocab_size [50000]** 
Size of the source vocabulary

* **-tgt_vocab_size [50000]** 
Size of the target vocabulary

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

### **Random**:
* **-shuffle [1]** 
Shuffle data

* **-seed [3435]** 
Random seed

### **Logging**:
* **-report_every [100000]** 
Report status every this many sentences

### **Speech**:
* **-sample_rate [16000]** 
Sample rate.

* **-window_size [0.02]** 
Window size for spectrogram in seconds.

* **-window_stride [0.01]** 
Window stride for spectrogram in seconds.

* **-window [hamming]** 
Window type for spectrogram generation.
