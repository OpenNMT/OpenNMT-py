<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

# translate.py:

```
usage: translate.py [-h] [-md] -model MODEL -src SRC
                    [-src_img_dir SRC_IMG_DIR] [-tgt TGT] [-output OUTPUT]
                    [-beam_size BEAM_SIZE] [-batch_size BATCH_SIZE]
                    [-max_sent_length MAX_SENT_LENGTH] [-replace_unk]
                    [-verbose] [-attn_debug] [-dump_beam DUMP_BEAM]
                    [-n_best N_BEST] [-gpu GPU]

```

translate.py

## **optional arguments**:
### **-h, --help** 

```
show this help message and exit
```

### **-md** 

```
print Markdown-formatted help text and exit.
```

### **-model MODEL** 

```
Path to model .pt file
```

### **-src SRC** 

```
Source sequence to decode (one line per sequence)
```

### **-src_img_dir SRC_IMG_DIR** 

```
Source image directory
```

### **-tgt TGT** 

```
True target sequence (optional)
```

### **-output OUTPUT** 

```
Path to output the predictions (each line will be the decoded sequence
```

### **-beam_size BEAM_SIZE** 

```
Beam size
```

### **-batch_size BATCH_SIZE** 

```
Batch size
```

### **-max_sent_length MAX_SENT_LENGTH** 

```
Maximum sentence length.
```

### **-replace_unk** 

```
Replace the generated UNK tokens with the source token that had highest
attention weight. If phrase_table is provided, it will lookup the identified
source token and give the corresponding target token. If it is not provided (or
the identified source token does not exist in the table) then it will copy the
source token
```

### **-verbose** 

```
Print scores and predictions for each sentence
```

### **-attn_debug** 

```
Print best attn for each word
```

### **-dump_beam DUMP_BEAM** 

```
File to dump beam information to.
```

### **-n_best N_BEST** 

```
If verbose is set, will output the n_best decoded sentences
```

### **-gpu GPU** 

```
Device to run on
```
