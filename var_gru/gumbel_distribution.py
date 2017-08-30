#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:32:56 2017

@author: tesla
Batched Gumbel SoftMax 
Adapt it to sequence modeling.

Paper:
    CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX
    https://arxiv.org/abs/1611.01144

    The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
    https://arxiv.org/abs/1611.00712

Code:
    https://github.com/ericjang/gumbel-softmax
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from collections import namedtuple


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probability distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


if __name__ == "__main__":
    sequence_length = 25
    vocab_size = 100
    hidden_dims = 10
    batch_size = 32

    output_logits = tf.constant(
        np.random.randn(batch_size, sequence_length, vocab_size), dtype=tf.float32)
    print("output logits shape", output_logits.get_shape())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        softmax = gumbel_softmax(output_logits, 1.0, hard=True)
        softmax_ = sess.run(softmax)
        print("softmax shape", softmax_.shape)
