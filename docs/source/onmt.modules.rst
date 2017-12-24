Doc: Modules
=============

Core Modules
------------

.. autoclass:: onmt.modules.Embeddings
    :members:


Encoders
---------

.. autoclass:: onmt.modules.EncoderBase
    :members:

.. autoclass:: onmt.modules.MeanEncoder
    :members:

.. autoclass:: onmt.modules.RNNEncoder
    :members:


Decoders
---------


.. autoclass:: onmt.modules.RNNDecoderBase
    :members:


.. autoclass:: onmt.modules.StdRNNDecoder
    :members:


.. autoclass:: onmt.modules.InputFeedRNNDecoder
    :members:

Attention
----------

.. autoclass:: onmt.modules.GlobalAttention
    :members:



Architecture: Transfomer
----------------------------

.. autoclass:: onmt.modules.PositionalEncoding
    :members:

.. autoclass:: onmt.modules.PositionwiseFeedForward
    :members:

.. autoclass:: onmt.modules.TransformerEncoder
    :members:

.. autoclass:: onmt.modules.TransformerDecoder
    :members:

.. autoclass:: onmt.modules.MultiHeadedAttention
    :members:
    :undoc-members:


Architecture: Conv2Conv
----------------------------

(These methods are from a user contribution
and have not been thoroughly tested.)


.. autoclass:: onmt.modules.CNNEncoder
    :members:


.. autoclass:: onmt.modules.CNNDecoder
    :members:

.. autoclass:: onmt.modules.ConvMultiStepAttention
    :members:

.. autoclass:: onmt.modules.WeightNorm
    :members:

Architecture: SRU
----------------------------

.. autoclass:: onmt.modules.SRU
    :members:


Alternative Encoders
--------------------

onmt\.modules\.AudioEncoder

.. autoclass:: onmt.modules.AudioEncoder
    :members:


onmt\.modules\.ImageEncoder

.. autoclass:: onmt.modules.ImageEncoder
    :members:


Copy Attention
--------------

.. autoclass:: onmt.modules.CopyGenerator
    :members:


Structured Attention
-------------------------------------------

.. autoclass:: onmt.modules.MatrixTree
    :members:
