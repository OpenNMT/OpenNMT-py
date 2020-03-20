Dataloader with dynamicly sampled noise and task-mix scheduling
===============================================================

We contribute a replacement for the OpenNMT-py dataloader,
with two new features: dynamicly sampled noise and task-mix scheduling.
The implementation is intended as a prototype to start a discussion
and hopefully to inform the design of a set of changes to the OpenNMT-py dataloader.
The code functions, but is not polished enough to be merged to master outright.

Why do we need sampling?
------------------------

The power of noise as a regularizer has been put to use in many methods advancing deep learning,
such as dropout (Srivastava et al., 2014), label smoothing (Szegedy et al., 2015), and SwitchOut (Wang et al., 2018).
Those three can be applied without changes to the dataloader.
However, recently methods resulting in much larger changes to the sequence,
such as subword regularization (Kudo, 2018), have been proposed.
In subword regularization, the way in which words are segmented into subwords is resampled each time the word occurs.
E.g. "unreasonable" might be segmented into "un + reasonable" once and "unreason + able" the next time, and "un + reason + able" a third time.
When converted into numeric indices into the vocabulary, these representations are very different.

In our recent paper (Grönroos et al., 2020a) seek to improve Neural Machine Translation (NMT)
into morphologically rich low-resource languages.
We use a very small amount of parallel data (10000 sentence pairs is enough),
but still reach a reasonable translation quality by making good use of
abundant monolingual data, and parallel data in a related high-resource language pair.

Our method relies on two new features that we implement into OpenNMT-py:

  1. For subword regularization and the denoising sequence autoencoder, 
     we need the ability to sample slightly different versions each time an example is used in training.
  2. For scheduled multi-task learning we need the ability to adjust the task mix during training.

The task-mix scheduling is important in multilingual settings when different language pairs have different amounts of data,
but can also be useful when mixing other types of data, such as
different domains (oversampling the in-domain data),
natural vs synthetic (e.g. back-translated) data,
and auxiliary tasks (e.g. autoencoder).

Current OpenNMT-py dataloader
-----------------------------

The current dataloader in OpenNMT-py is divided into two parts: preprocessing and training.
During preprocessing, data is divided into shards that fit into memory.
For each shard, a torchtext Dataset is created, hacked a bit, and saved to disk as a pickle.
During training, these shards are loaded, numericalized, divided into minibatches, and padded to the same length,
after which they are ready to be used.

A large amount of work must be done externally prior to running the preprocessing

    1. Cleaning and normalization.
    2. Pretokenization and subword segmentation.
    3. Oversampling data to achieve the desired mix (recently a new feature was introduced which allows a constant oversampling rate to be specified during preprocessing).
    4. Shuffling the data.

The current dataloader is to some extent an abuse of the torchtext library.
This is partly due to bad design choices in torchtext, which make correct usage difficult and unintuitive.
E.g. torchtext doesn't support non-toy-sized datasets that don't fit in memory at once,
necessitating users of the library to write their own sharding solutions.

Pickling Dataset objects is not an elegant solution, and doesn't accomplish very much.
When written to disk, the data is tokenized, but still in symbolic (non-numericalized) form.
There is some speed benefit over the use of plain text files,
as the binary format is faster to read (no need to scan for newlines), and the cost of tokenization is paid in advance.

Unfortunately there are many downsides.
Every variation needs a separate preprocessing run, which takes up a lot of disk space.
The problem is particulary severe for researchers doing experiments on different kinds of preprocessing, e.g. subword segmentation.
In one of my experiments I had over a terabyte of redundant oversampled variants of the same data with different preprocessing.
A constant mixing for corpora was recently introduced, but before that oversampling had to be done in preprocessing.

Proposed alternative
--------------------

### Concepts

**Input corpora**.
Multiple input corpora can be used.
During offline preprocessing, the corpora can be kept separated: there is no need to concatenate them into mixed files.
As the transform processing pipeline is very powerful, the input corpora can stay as (or close to) raw text.
In particular any variant processing steps that you want to experiment with should not be necessary to pre-apply offline.

**Tasks** (Note that these are called "groups" in the current implementation).
A task consists of a particular type of data that should all be treated in the same way.

  - In multilingual training, different language pairs are separate tasks.
  - To treat back-translated synthetic data differently from natural data (e.g. weight it differenly or prepend a special token),
    the two are made into separate tasks.
  - Adding an autoencoder auxiliary task requires a processing pipeline that is different from the main task.

A task can use data from multiple input corpora, in which case they are mixed together during sharding,
in such a way that each shard contains the same mix of input corpora.
This avoids situations where shards are imbalanced, leading to a long time training on a single corpus.
Examples from different tasks are mixed together during training, using a time-varying task-mix schedule.

Tasks are either parallel or monolingual.
This determines the type of the input corpora, either a separate file for source and target or a single file.
After the processing pipeline is finished, examples from both types of task consist of a source and target side.
More task types could be defined, e.g. for multimodal NMT.

Tasks belong to either the training or the validation split.
Processing for the validation data is controllable in the same way as for training.

**Transforms**
The processing pipeline consists of a series of transforms.
The transforms can modify the data in powerful ways:
make arbitrary changes to the sequence of tokens (including changing its length),
duplicate monolingual data into a source and target side,
or even filter out examples.

### Config files

The data and the transforms that should be applied to it are defined in separate dataloader configs, rather than as command line options.
Command line options with the same flexibility would become very complex, and thus difficult to read and modify.

An example sharding config can be found in `examples/dynamicdata.shard.ural.lowres18k.bt.yaml`.
An example training config can be found in `examples/dynamicdata.train.ural.lowres18k.bt.jtokemprune16k.faster.mono3.noise2.seq2.yaml`.

Note that the sharding config is a subset of the train config.
The common parts need to match exactly.
If only one training run is made, the training config can be used directly as the sharding config (the extra values are ignored).
If the same sharding is used in many training runs, the separation of configs is necessary.

#### Some details about the configs

`meta.shard.pretokenize` and `meta.shard.predetokenize`: whether to run pyonmttok (or undo tokenization, for SentencePiece) when sharding. 
This flexibility allows easily using either raw untokenized corpora and corpora that have been pretokenized because some offline preprocessing step needs it.

`meta.train.name` determines where the transforms are saved. It is possible to use the same transforms with different mixing weights by making another training conf using the same name parameter. Ususally it should be unique, though.

`meta.train.mixing_weight_schedule` determines after which number of minibatches the mixing weights should be adjusted. The `groups.*.weight` parameters should be of length one longer than this.

`meta.train.*` global parameters for the transforms, e.g. setting the amount of noise.

`groups.*.meta.src_lang` and `groups.*.meta.trg_lang` are used by the `lang_prefix_both` transform to produce (target) language tokens. `groups.*.meta.extra_prefix` can be used to mark synthetic data, such as in the back-translation group `enet_bt` in the example.

`groups.*.n_shards` allows overriding the number of shards for a single task. Useful if a task is low-resource.

`groups.*.share_inputs` allows a task to use the same inputs as another task, without the need to shard the data twice. This makes it possible to apply e.g. two different autoencoder tasks to the same monolingual data. In the example `mono_fi_taboo` has `share_inputs: mono_fi`. No inputs should be assigned directly to a task that uses `share_inputs`. 

### Usage

    0. **Offline preprocessing:** e.g. cleaning, pretokenization.
       This is an optional step. For computational reasons, anything non-stochastic that is not varied between experiments can be done offline in advance.
    1. **Sharding.** `preprocess_dynamicdata.py` uses the sharding config. The input corpora for each task are read, mixed together and divided into shards, in such a way that each shard gets the same mix of input corpora. The shards are plain text files. At the same time, a word vocabulary is computed.
    2. **Train segmentation model, determine vocabulary.** The segmentation model is trained in the same way as previously. The word vocabulary computed in the previous step can be used. It is important to determine all subwords that the segmentation model might use, in order to determine the NMT model vocabulary.
    3. **Setting up transforms.** The torchtext Field objects are created. Transforms are warmed up, e.g. precomputing a cache of segmentation alternatives for the most common words. This is currently important for translation speed (although saving the transforms in the model checkpoint file would solve this better).
    4. **Training.** `train_dynamicdata.py` uses the training conf.

### Example

![Preprocessing and training steps](dyndata.png)
![Example transforms](transforms.png)

### Under the hood

#### How the mixing works

During sharding vs during training. During sharding: corpora within each task are divided evenly into the shards. This is a constant mix without oversampling.
During training: tasks are mixed according to the task-mix weight schedule. A single minibatch can contain examples from many different tasks.

Note that the task-mix weight schedule currently counts data minibatches, not parameter updates.
This is relevant when using gradient accumulation or multi-GPU training.
E.g. with gradient accumulated over 4 minibatches, to change mixing distribution after 40k parameter updates the training conf mixing weight schedule needs to be set to [160000].

### Heavy processing in the data loader

Some transforms, such as subword segmentation, may involve heavy computation.
To ensure that the GPU is kept at maximum capacity,
we move the dataloader to a separate process and use a torch.multiprocessing queue to communicate the minibatches to the trainer process. This setup is already implemented in OpenNMT-py for multi-GPU training.

(Note: our cluster is configured with the GPU compute mode set to "Exclusive Process", which means that only one process can access the GPU. Multi-threading would work, but not multi-process. To ensure that only the trainer process accesses the GPU, the current implementation transfers minibatches as CPU tensors, which are sent to the GPU by the trainer. This is less efficient than sending them in the data loader, after which the inter-process communication is very light.)


Potential improvements
----------------------

References
----------

Grönroos SA, Virpioja S, Kurimo M (2020a)
    Transfer learning and subword sampling for asymmetric-resource one-to-many neural translation. In review.

Grönroos SA, Virpioja S, Kurimo M (2020b)
    Morfessor EM+Prune: Improved subword segmentation with expectation maximization and pruning.
    In: Proceedings of the 12th Language Resources and Evaluation Conference, ELRA, Marseilles, France, to appear
    [arXiv: 2003.03131](https://arxiv.org/abs/2003.03131)

Kudo T (2018)
     Subword regularization: Improving neural network translation models with multiple subword candidates. [arXiv: 1804.10959](http://arxiv.org/abs/1804.10959)

Srivastava N, Hinton G, Krizhevsky A, Sutskever I, Salakhutdinov R (2014)
    [Dropout: a simple way to prevent neural networks from overfitting.](https://dl.acm.org/doi/abs/10.5555/2627435.2670313) The Journal of Machine Learning Research 15(1):1929–1958

Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z (2015)
    Rethinking the inception architecture for computer vision. [arXiv: 1512.00567](http://arxiv.org/abs/1512.00567)

Wang X, Pham H, Dai Z, Neubig G (2018)
    SwitchOut: an efficient data augmentation algorithm for neural machine translation. [arXiv: 1808.07512](https://arxiv.org/abs/1808.07512)
