<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

# train.py:

```
usage: train.py [-h] [-md] -data DATA [-save_model SAVE_MODEL]
                [-train_from_state_dict TRAIN_FROM_STATE_DICT]
                [-train_from TRAIN_FROM] [-layers LAYERS] [-rnn_size RNN_SIZE]
                [-word_vec_size WORD_VEC_SIZE] [-input_feed INPUT_FEED]
                [-brnn] [-brnn_merge BRNN_MERGE] [-batch_size BATCH_SIZE]
                [-max_generator_batches MAX_GENERATOR_BATCHES]
                [-epochs EPOCHS] [-start_epoch START_EPOCH]
                [-param_init PARAM_INIT] [-optim OPTIM]
                [-max_grad_norm MAX_GRAD_NORM] [-dropout DROPOUT]
                [-curriculum] [-extra_shuffle] [-learning_rate LEARNING_RATE]
                [-learning_rate_decay LEARNING_RATE_DECAY]
                [-start_decay_at START_DECAY_AT]
                [-pre_word_vecs_enc PRE_WORD_VECS_ENC]
                [-pre_word_vecs_dec PRE_WORD_VECS_DEC] [-gpus GPUS [GPUS ...]]
                [-log_interval LOG_INTERVAL]

```

train.py

## **optional arguments**:
### **-h, --help** 

```
show this help message and exit
```

### **-md** 

```
print Markdown-formatted help text and exit.
```

### **-data DATA** 

```
Path to the *-train.pt file from preprocess.py
```

### **-save_model SAVE_MODEL** 

```
Model filename (the model will be saved as <save_model>_epochN_PPL.pt where PPL
is the validation perplexity
```

### **-train_from_state_dict TRAIN_FROM_STATE_DICT** 

```
If training from a checkpoint then this is the path to the pretrained model's
state_dict.
```

### **-train_from TRAIN_FROM** 

```
If training from a checkpoint then this is the path to the pretrained model.
```

### **-layers LAYERS** 

```
Number of layers in the LSTM encoder/decoder
```

### **-rnn_size RNN_SIZE** 

```
Size of LSTM hidden states
```

### **-word_vec_size WORD_VEC_SIZE** 

```
Word embedding sizes
```

### **-input_feed INPUT_FEED** 

```
Feed the context vector at each time step as additional input (via concatenation
with the word embeddings) to the decoder.
```

### **-brnn** 

```
Use a bidirectional encoder
```

### **-brnn_merge BRNN_MERGE** 

```
Merge action for the bidirectional hidden states: [concat|sum]
```

### **-batch_size BATCH_SIZE** 

```
Maximum batch size
```

### **-max_generator_batches MAX_GENERATOR_BATCHES** 

```
Maximum batches of words in a sequence to run the generator on in parallel.
Higher is faster, but uses more memory.
```

### **-epochs EPOCHS** 

```
Number of training epochs
```

### **-start_epoch START_EPOCH** 

```
The epoch from which to start
```

### **-param_init PARAM_INIT** 

```
Parameters are initialized over uniform distribution with support (-param_init,
param_init)
```

### **-optim OPTIM** 

```
Optimization method. [sgd|adagrad|adadelta|adam]
```

### **-max_grad_norm MAX_GRAD_NORM** 

```
If the norm of the gradient vector exceeds this, renormalize it to have the norm
equal to max_grad_norm
```

### **-dropout DROPOUT** 

```
Dropout probability; applied between LSTM stacks.
```

### **-curriculum** 

```
For this many epochs, order the minibatches based on source sequence length.
Sometimes setting this to 1 will increase convergence speed.
```

### **-extra_shuffle** 

```
By default only shuffle mini-batch order; when true, shuffle and re-assign mini-
batches
```

### **-learning_rate LEARNING_RATE** 

```
Starting learning rate. If adagrad/adadelta/adam is used, then this is the
global learning rate. Recommended settings: sgd = 1, adagrad = 0.1, adadelta =
1, adam = 0.001
```

### **-learning_rate_decay LEARNING_RATE_DECAY** 

```
If update_learning_rate, decay learning rate by this much if (i) perplexity does
not decrease on the validation set or (ii) epoch has gone past start_decay_at
```

### **-start_decay_at START_DECAY_AT** 

```
Start decaying every epoch after and including this epoch
```

### **-pre_word_vecs_enc PRE_WORD_VECS_ENC** 

```
If a valid path is specified, then this will load pretrained word embeddings on
the encoder side. See README for specific formatting instructions.
```

### **-pre_word_vecs_dec PRE_WORD_VECS_DEC** 

```
If a valid path is specified, then this will load pretrained word embeddings on
the decoder side. See README for specific formatting instructions.
```

### **-gpus GPUS [GPUS ...]** 

```
Use CUDA on the listed devices.
```

### **-log_interval LOG_INTERVAL** 

```
Print stats at this interval.
```
