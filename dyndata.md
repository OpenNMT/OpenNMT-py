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

Proposed alternative
--------------------

![Preprocessing and training steps](dyndata.png)

### Usage

### Under the hood

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
