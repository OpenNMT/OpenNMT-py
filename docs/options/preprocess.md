<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

# preprocess.py:

```
usage: preprocess.py [-h] [-md] [-config CONFIG] [-src_type SRC_TYPE]
                     [-src_img_dir SRC_IMG_DIR] -train_src TRAIN_SRC
                     -train_tgt TRAIN_TGT -valid_src VALID_SRC -valid_tgt
                     VALID_TGT -save_data SAVE_DATA
                     [-src_vocab_size SRC_VOCAB_SIZE]
                     [-tgt_vocab_size TGT_VOCAB_SIZE] [-src_vocab SRC_VOCAB]
                     [-tgt_vocab TGT_VOCAB]
                     [-features_vocabs_prefix FEATURES_VOCABS_PREFIX]
                     [-src_seq_length SRC_SEQ_LENGTH]
                     [-src_seq_length_trunc SRC_SEQ_LENGTH_TRUNC]
                     [-tgt_seq_length TGT_SEQ_LENGTH]
                     [-tgt_seq_length_trunc TGT_SEQ_LENGTH_TRUNC]
                     [-shuffle SHUFFLE] [-seed SEED] [-lower]
                     [-report_every REPORT_EVERY]

```

preprocess.py

## **optional arguments**:
### **-h, --help** 

```
show this help message and exit
```

### **-md** 

```
print Markdown-formatted help text and exit.
```

### **-config CONFIG** 

```
Read options from this file
```

### **-src_type SRC_TYPE** 

```
Type of the source input. Options are [text|img].
```

### **-src_img_dir SRC_IMG_DIR** 

```
Location of source images
```

### **-train_src TRAIN_SRC** 

```
Path to the training source data
```

### **-train_tgt TRAIN_TGT** 

```
Path to the training target data
```

### **-valid_src VALID_SRC** 

```
Path to the validation source data
```

### **-valid_tgt VALID_TGT** 

```
Path to the validation target data
```

### **-save_data SAVE_DATA** 

```
Output file for the prepared data
```

### **-src_vocab_size SRC_VOCAB_SIZE** 

```
Size of the source vocabulary
```

### **-tgt_vocab_size TGT_VOCAB_SIZE** 

```
Size of the target vocabulary
```

### **-src_vocab SRC_VOCAB** 

```
Path to an existing source vocabulary
```

### **-tgt_vocab TGT_VOCAB** 

```
Path to an existing target vocabulary
```

### **-features_vocabs_prefix FEATURES_VOCABS_PREFIX** 

```
Path prefix to existing features vocabularies
```

### **-src_seq_length SRC_SEQ_LENGTH** 

```
Maximum source sequence length
```

### **-src_seq_length_trunc SRC_SEQ_LENGTH_TRUNC** 

```
Truncate source sequence length.
```

### **-tgt_seq_length TGT_SEQ_LENGTH** 

```
Maximum target sequence length to keep.
```

### **-tgt_seq_length_trunc TGT_SEQ_LENGTH_TRUNC** 

```
Truncate target sequence length.
```

### **-shuffle SHUFFLE** 

```
Shuffle data
```

### **-seed SEED** 

```
Random seed
```

### **-lower** 

```
lowercase data
```

### **-report_every REPORT_EVERY** 

```
Report status every this many sentences
```
