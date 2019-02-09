<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

translate.py
# Options: translate.py:
translate.py Args that start with '--' (eg. --md) can also be set in a config
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

### **Model**:
* **-models []** 
Path to model .pt file(s). Multiple models can be specified, for ensemble
decoding.

* **-avg_raw_probs []** 
If this is set, during ensembling scores from different models will be combined
by averaging their raw probabilities and then taking the log. Otherwise, the log
probabilities will be averaged directly. Necessary for models whose output
layers can assign zero probability.

### **Data**:
* **-data_type [text]** 
Type of the source input. Options: [text|img].

* **-src []** 
Source sequence to decode (one line per sequence)

* **-src_dir []** 
Source directory for image or audio files

* **-tgt []** 
True target sequence (optional)

* **-shard_size [10000]** 
Divide src and tgt (if applicable) into smaller multiple src and tgt files, then
build shards, each shard will have opt.shard_size samples except last shard.
shard_size=0 means no segmentation shard_size>0 means segment dataset into
multiple shards, each shard has shard_size samples

* **-output [pred.txt]** 
Path to output the predictions (each line will be the decoded sequence

* **-report_bleu []** 
Report bleu score after translation, call tools/multi-bleu.perl on command line

* **-report_rouge []** 
Report rouge 1/2/3/L/SU4 score after translation call tools/test_rouge.py on
command line

* **-report_time []** 
Report some translation time metrics

* **-dynamic_dict []** 
Create dynamic dictionaries

* **-share_vocab []** 
Share source and target vocabulary

### **Random Sampling**:
* **-random_sampling_topk [1]** 
Set this to -1 to do random sampling from full distribution. Set this to value
k>1 to do random sampling restricted to the k most likely next tokens. Set this
to 1 to use argmax or for doing beam search.

* **-random_sampling_temp [1.0]** 
If doing random sampling, divide the logits by this before computing softmax
during decoding.

* **-seed [829]** 
Random seed

### **Beam**:
* **-beam_size [5]** 
Beam size

* **-min_length []** 
Minimum prediction length

* **-max_length [100]** 
Maximum prediction length.

* **-max_sent_length []** 
Deprecated, use `-max_length` instead

* **-stepwise_penalty []** 
Apply penalty at every decoding step. Helpful for summary penalty.

* **-length_penalty [none]** 
Length Penalty to use.

* **-coverage_penalty [none]** 
Coverage Penalty to use.

* **-alpha []** 
Google NMT length penalty parameter (higher = longer generation)

* **-beta []** 
Coverage penalty parameter

* **-block_ngram_repeat []** 
Block repetition of ngrams during decoding.

* **-ignore_when_blocking []** 
Ignore these strings when blocking repeats. You want to block sentence
delimiters.

* **-replace_unk []** 
Replace the generated UNK tokens with the source token that had highest
attention weight. If phrase_table is provided, it will lookup the identified
source token and give the corresponding target token. If it is not provided(or
the identified source token does not exist in the table) then it will copy the
source token

### **Logging**:
* **-verbose []** 
Print scores and predictions for each sentence

* **-log_file []** 
Output logs to a file under this path.

* **-log_file_level [0]** 

* **-attn_debug []** 
Print best attn for each word

* **-dump_beam []** 
File to dump beam information to.

* **-n_best [1]** 
If verbose is set, will output the n_best decoded sentences

### **Efficiency**:
* **-batch_size [30]** 
Batch size

* **-gpu [-1]** 
Device to run on

### **Speech**:
* **-sample_rate [16000]** 
Sample rate.

* **-window_size [0.02]** 
Window size for spectrogram in seconds

* **-window_stride [0.01]** 
Window stride for spectrogram in seconds

* **-window [hamming]** 
Window type for spectrogram generation

* **-image_channel_size [3]** 
Using grayscale image can training model faster and smaller
