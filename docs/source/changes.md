

# Versions

**OpenNMT-py v3 release **

This new version does not rely on Torchtext anymore.
The checkpoint structure is slightly changed but we provide a tool to convert v2 to v3 models (cf tools/convertv2_v3.py)

We use the same 'dynamic' paradigm as in v2, allowing to apply on-the-fly transforms to the data.

This has a few advantages, amongst which:

- remove or drastically reduce the preprocessing required to train a model;
- increase the possibilities of data augmentation and manipulation through on-the fly transforms.

These transforms can be specific tokenization methods, filters, noising, or any custom transform users may want to implement. Custom transform implementation is quite straightforward thanks to the existing base class and example implementations.

You can check out how to use this new data loading pipeline in the updated [docs](https://opennmt.net/OpenNMT-py).

All the readily available transforms are described [here](https://opennmt.net/OpenNMT-py/FAQ.html#what-are-the-readily-available-on-the-fly-data-transforms).

### Breaking changes

Changes between v2 and v3:

Options removed:
`queue_size`, `pool_factor` are no longer needed. Only adjust the `bucket_size` to the number of examples to be loaded by each `num_workers` of the pytorch Dataloader.

New options: 
`num_workers`: number of workers for each process. If you run on one GPU the recommended value is 4. If you run on more than 1 GPU, the recommended value is 2
`add_qkvbias`: default is false. However old model trained with v2 will be set at true. The original transformer paper used no bias for the Q/K/V nn.Linear of the multihead attention module.

Options renamed:
`rnn_size` => `hidden_size`
`enc_rnn_size` => `enc_hid_size`
`dec_rnn_size` => `dec_hid_size`

Note: `tools/convertv2_v3.py` will modify these options stored in the checkpoint to make things compatible with v3.0

Inference:
The translator will use the same dynamic_iterator as the trainer.
The new default for inference is `length_penalty=avg` which will provide better BLEU scores in most cases (and comparable to other toolkits defaults)

Reminder: a few features were dropped between v1 and v2:

- audio, image and video inputs;

For any user that still need these features, the previous codebase will be retained as `legacy` in a separate branch. It will no longer receive extensive development from the core team but PRs may still be accepted.

Feel free to check it out and let us know what you think of the new paradigm!

### Performance tips

Given sufficient CPU resources according to GPU computing power, most of the transforms should not slow the training down. (Note: for now, one producer process per GPU is spawned -- meaning you would ideally need 2N CPU threads for N GPUs).
If you want to optimize the training performance:
- use fp16
- use batch_size_multiple 8
- use vocab_size_multiple 8
- Depending on the number of GPU use num_workers 4 (for 1 GPU) or 2 (for multiple GPU)

- To avoid averaging checkpoints you can use the "during training" average decay system.
- If you train a transformer we support max_relative_positions (use 20) instead of position_encoding.

- for very fast inference convert your model to [CTranslate2](https://github.com/OpenNMT/CTranslate2) format. 
