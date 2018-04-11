<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->

train.py
# Options: train.py:
train.py

### **Model-Embeddings**:
* **-src_word_vec_size [500]** 
Word embedding size for src.

* **-tgt_word_vec_size [500]** 
Word embedding size for tgt.

* **-word_vec_size [-1]** 
Word embedding size for src and tgt.

* **-share_decoder_embeddings []** 
Use a shared weight matrix for the input and output word embeddings in the
decoder.

* **-share_embeddings []** 
Share the word embeddings between encoder and decoder. Need to use shared
dictionary for this option.

* **-position_encoding []** 
Use a sin to mark relative words positions. Necessary for non-RNN style models.

### **Model-Embedding Features**:
* **-feat_merge [concat]** 
Merge action for incorporating features embeddings. Options [concat|sum|mlp].

* **-feat_vec_size [-1]** 
If specified, feature embedding sizes will be set to this. Otherwise,
feat_vec_exponent will be used.

* **-feat_vec_exponent [0.7]** 
If -feat_merge_size is not set, feature embedding sizes will be set to
N^feat_vec_exponent where N is the number of values the feature takes.

### **Model- Encoder-Decoder**:
* **-model_type [text]** 
Type of source model to use. Allows the system to incorporate non-text inputs.
Options are [text|img|audio].

* **-encoder_type [rnn]** 
Type of encoder layer to use. Non-RNN layers are experimental. Options are
[rnn|brnn|mean|transformer|cnn].

* **-decoder_type [rnn]** 
Type of decoder layer to use. Non-RNN layers are experimental. Options are
[rnn|transformer|cnn].

* **-layers [-1]** 
Number of layers in enc/dec.

* **-enc_layers [2]** 
Number of layers in the encoder

* **-dec_layers [2]** 
Number of layers in the decoder

* **-rnn_size [500]** 
Size of rnn hidden states

* **-cnn_kernel_width [3]** 
Size of windows in the cnn, the kernel_size is (cnn_kernel_width, 1) in conv
layer

* **-input_feed [1]** 
Feed the context vector at each time step as additional input (via concatenation
with the word embeddings) to the decoder.

* **-bridge []** 
Have an additional layer between the last encoder state and the first decoder
state

* **-rnn_type [LSTM]** 
The gate type to use in the RNNs

* **-brnn []** 
Deprecated, use `encoder_type`.

* **-brnn_merge [concat]** 
Merge action for the bidir hidden states

* **-context_gate []** 
Type of context gate to use. Do not select for no context gate.

### **Model- Attention**:
* **-global_attention [general]** 
The attention type to use: dotprod or general (Luong) or MLP (Bahdanau)

* **-copy_attn []** 
Train copy attention layer.

* **-copy_attn_force []** 
When available, train to copy.

* **-reuse_copy_attn []** 
Reuse standard attention for copy

* **-copy_loss_by_seqlength []** 
Divide copy loss by length of sequence

* **-coverage_attn []** 
Train a coverage attention layer.

* **-lambda_coverage [1]** 
Lambda value for coverage.

### **General**:
* **-data []** 
Path prefix to the ".train.pt" and ".valid.pt" file path from preprocess.py

* **-save_model [model]** 
Model filename (the model will be saved as <save_model>_epochN_PPL.pt where PPL
is the validation perplexity

* **-gpuid []** 
Use CUDA on the listed devices.

* **-seed [-1]** 
Random seed used for the experiments reproducibility.

### **Initialization**:
* **-start_epoch [1]** 
The epoch from which to start

* **-param_init [0.1]** 
Parameters are initialized over uniform distribution with support (-param_init,
param_init). Use 0 to not use initialization

* **-param_init_glorot []** 
Init parameters with xavier_uniform. Required for transfomer.

* **-train_from []** 
If training from a checkpoint then this is the path to the pretrained model's
state_dict.

* **-pre_word_vecs_enc []** 
If a valid path is specified, then this will load pretrained word embeddings on
the encoder side. See README for specific formatting instructions.

* **-pre_word_vecs_dec []** 
If a valid path is specified, then this will load pretrained word embeddings on
the decoder side. See README for specific formatting instructions.

* **-fix_word_vecs_enc []** 
Fix word embeddings on the encoder side.

* **-fix_word_vecs_dec []** 
Fix word embeddings on the encoder side.

### **Optimization- Type**:
* **-batch_size [64]** 
Maximum batch size for training

* **-batch_type [sents]** 
Batch grouping for batch_size. Standard is sents. Tokens will do dynamic
batching

* **-normalization [sents]** 
Normalization method of the gradient.

* **-accum_count [1]** 
Accumulate gradient this many times. Approximately equivalent to updating
batch_size * accum_count batches at once. Recommended for Transformer.

* **-valid_batch_size [32]** 
Maximum batch size for validation

* **-max_generator_batches [32]** 
Maximum batches of words in a sequence to run the generator on in parallel.
Higher is faster, but uses more memory.

* **-epochs [13]** 
Number of training epochs

* **-optim [sgd]** 
Optimization method.

* **-adagrad_accumulator_init []** 
Initializes the accumulator values in adagrad. Mirrors the
initial_accumulator_value option in the tensorflow adagrad (use 0.1 for their
default).

* **-max_grad_norm [5]** 
If the norm of the gradient vector exceeds this, renormalize it to have the norm
equal to max_grad_norm

* **-dropout [0.3]** 
Dropout probability; applied in LSTM stacks.

* **-truncated_decoder []** 
Truncated bptt.

* **-adam_beta1 [0.9]** 
The beta1 parameter used by Adam. Almost without exception a value of 0.9 is
used in the literature, seemingly giving good results, so we would discourage
changing this value from the default without due consideration.

* **-adam_beta2 [0.999]** 
The beta2 parameter used by Adam. Typically a value of 0.999 is recommended, as
this is the value suggested by the original paper describing Adam, and is also
the value adopted in other frameworks such as Tensorflow and Kerras, i.e. see:
https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
https://keras.io/optimizers/ . Whereas recently the paper "Attention is All You
Need" suggested a value of 0.98 for beta2, this parameter may not work well for
normal models / default baselines.

* **-label_smoothing []** 
Label smoothing value epsilon. Probabilities of all non-true labels will be
smoothed by epsilon / (vocab_size - 1). Set to zero to turn off label smoothing.
For more detailed information, see: https://arxiv.org/abs/1512.00567

### **Optimization- Rate**:
* **-learning_rate [1.0]** 
Starting learning rate. Recommended settings: sgd = 1, adagrad = 0.1, adadelta =
1, adam = 0.001

* **-learning_rate_decay [0.5]** 
If update_learning_rate, decay learning rate by this much if (i) perplexity does
not decrease on the validation set or (ii) epoch has gone past start_decay_at

* **-start_decay_at [8]** 
Start decaying every epoch after and including this epoch

* **-start_checkpoint_at []** 
Start checkpointing every epoch after and including this epoch

* **-decay_method []** 
Use a custom decay rate.

* **-warmup_steps [4000]** 
Number of warmup steps for custom decay.

### **Logging**:
* **-report_every [50]** 
Print stats at this interval.

* **-exp_host []** 
Send logs to this crayon server.

* **-exp []** 
Name of the experiment for logging.

* **-tensorboard []** 
Use tensorboardX for visualization during training. Must have the library
tensorboardX.

* **-tensorboard_log_dir [runs/onmt]** 
Log directory for Tensorboard. This is also the name of the run.

### **Speech**:
* **-sample_rate [16000]** 
Sample rate.

* **-window_size [0.02]** 
Window size for spectrogram in seconds.
