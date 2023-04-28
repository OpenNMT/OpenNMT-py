# Gated Graph Neural Networks

Graph-to-sequence networks allow information representable as a graph (such as an annotated NLP sentence or computer code structured as an AST) to be connected to a sequence generator to produce output which can benefit from the graph structure of the input.

The training option `-encoder_type ggnn` implements a GGNN (Gated Graph Neural Network) based on github.com/JamesChuanggg/ggnn.pytorch.git which is based on the paper "Gated Graph Sequence Neural Networks" by Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel.

The ggnn encoder is used for program equivalence proof generation in the paper [Equivalence of Dataflow Graphs via Rewrite Rules Using a Graph-to-Sequence Neural Model](https://arxiv.org/abs/2002.06799). That paper shows the benefit of the graph-to-sequence model over a sequence-to-sequence model for this problem which can be well represented with graphical input. The integration of the ggnn network into the OpenNMT-py system supports attention on the nodes as well as a copy mechanism.

### Dependencies

* There are no additional dependencies beyond the rnn-to-rnn sequence2sequence requirements.

### Quick Start

To get started, we provide a toy graph-to-sequence example. We assume that the working directory is `OpenNMT-py` throughout this document.

0) Download the data to a sibling directory.

```
cd ..
git clone https://github.com/SteveKommrusch/OpenNMT-py-ggnn-example
source OpenNMT-py-ggnn-example/env.sh
cd OpenNMT-py
```


The YAML configuration for this example is the following:

```yaml
# save_data is where the necessary objects will be written
save_data: OpenNMT-py-ggnn-example/run/example

# Filter long examples
src_seq_length: 1000
tgt_seq_length: 30

# Data definition
data:
    cnndm:
        path_src: OpenNMT-py-ggnn-example/src-train.txt
        path_tgt: OpenNMT-py-ggnn-example/tgt-train.txt
        transforms: [filtertoolong]
        weight: 1
    valid:
        path_src: OpenNMT-py-ggnn-example/src-val.txt
        path_tgt: OpenNMT-py-ggnn-example/tgt-val.txt

src_vocab: OpenNMT-py-ggnn-example/srcvocab.txt
tgt_vocab: OpenNMT-py-ggnn-example/tgtvocab.txt

save_model: OpenNMT-py-ggnn-example/run/model

# Model options
train_steps: 10000
save_checkpoint_steps: 5000
encoder_type: ggnn
layers: 2
decoder_type: rnn
learning_rate: 0.1
start_decay_steps: 5000
learning_rate_decay: 0.8
global_attention: general
batch_size: 32
# src_ggnn_size is larger than vocab plus features to allow one-hot settings
src_ggnn_size: 100
# src_word_vec_size less than hidden_size allows rnn learning during GGNN steps
src_word_vec_size: 16
# Increase tgt_word_vec_size, hidden_size, and state_dim together
# to provide larger GGNN embeddings and larger decoder RNN
tgt_word_vec_size: 64
hidden_size: 64
state_dim: 64
bridge: true
gpu_ranks: 0
n_edge_types: 9
# Increasing n_steps slows model computation but allows information
# to be aggregated over more node hops
n_steps: 5
n_node: 70
```

2) Train the model.

```
python train.py -config examples/ggnn.yaml
```

3) Translate the graph of 2 equivalent linear algebra expressions into the axiom list which proves them equivalent.

```
python translate.py \
    -model ../OpenNMT-py-ggnn-example/run/model_step_10000.pt \
    -src ../OpenNMT-py-ggnn-example/src-test.txt \
    -beam_size 5 -n_best 5 \
    -gpu 0 \
    -output ../OpenNMT-py-ggnn-example/run/pred-test_beam5.txt \
    > ../OpenNMT-py-ggnn-example/run/translate5.out 2>&1
```
### Graph data format

The GGNN implementation leverages the sequence processing and vocabulary
interface of OpenNMT. Each graph is provided on an input line, much like
a sentence is provided on an input line. A graph nearal network input line
includes `sentence tokens`, `feature values`, and `edges` separated by
`<EOT>` (end of tokens) tokens. Below is example of the input for a pair
of algebraic equations structured as a graph:

```
Sentence tokens       Feature values           Edges
---------------       ------------------       -------------------------------------------------------
- - - 0 a a b b <EOT> 0 1 2 3 4 4 2 3 12 <EOT> 0 2 1 3 2 4 , 0 6 1 7 2 5 , 0 4 , 0 5 , , , , 8 0 , 8 1
```

The equations being represented are `((a - a) - b)` and `(0 - b)`, the 
`sentence tokens` are provided before the first `<EOT>`. After
the first `<EOT>`, the `features values` are provided. These are extra
flags with information on each node in the graph. In this case, the 8
sentence tokens have feature flags ranging from 0 to 4; the 9th feature
flag defines a 9th node in the graph which does not have sentence token
information, just feature data. Nodes with any non-number flag (such as
`-` or `.`) will not have a feature added. Multiple groups of features
can be provided by using the `,` delimiter between the first and second
'<EOT>' tokens. After the second `<EOT>` token, edge information is provided.
Edge data is given as node pairs, hence `<EOT> 0 2 1 3` indicates that there
are edges from node 0 to node 2 and from node 1 to node 3. The GGNN supports
multiple edge types (which result mathematically in multiple weight matrices
for the model) and the edge types are separated by `,` tokens after the
second `<EOT>` token.

Note that the source vocabulary file needs to include the '<EOT>' token,
the ',' token, and all of the numbers used for feature flags and node
identifiers in the edge list.

### Vocabulary notes

Because edge information and feature data is provided through tokens in the source files, the `-src_vocab` file requires a certain format. The `<EOT>` token should occur in the vocab files after all tokens which are part of the `sentence tokens` shown above. After the `<EOT>` token, any remaining numerical tokens appropriate for node numbers or feature values can be included too (it is OK for integers to occur in the `sentence tokens` and such tokens should not be duplicated after the `<EOT>` token). The full size of the vector used as input per node is the number of tokens up to and including `<EOT>` plus the largest feature number used in the input. If the optional `src_ggnn_size` parameter is used to create an embedding layer, then its value must be at or above the full node input vector size; the embedding initializes the lower `src_word_vec_size` dimensions of the node value. If `src_ggnn_size` is not used, then `state_dim` must bet at or above the full node input vector size; as there is no embedding layer in this case, the initial value of the node is set directly.
Generally, one can use `onmt_build_vocab` to process GGNN input data to create vocab files and then adjust the source vocab appropriately. For an example of generating and adjusting a vocabulary for GGNN, please refer to [GGNN end-to-end example](https://github.com/SteveKommrusch/OpenNMT-py-ggnn-example#graph-input-processing-end-to-end-example).

### Options

* `-rnn_type (str)`: style of recurrent unit to use, one of [LSTM]
* `src_ggnn_size (int)`: Size of token-to-node embedding input
* `src_word_vec_size (int)`: Size of token-to-node embedding output
* `-state_dim (int)`: Number of state dimensions in nodes
* `-n_edge_types (int)`: Number of edge types
* `-bidir_edges (bool)`: True if reverse edges should be automatically created
* `-n_node (int)`: Max nodes in graph
* `-bridge_extra_node (bool)`: True indicates only the vector from the 1st extra node (after token listing) should be used for decoder initialization; False indicates all node vectors should be averaged together for decoder initialization
* `-n_steps (int)`: Steps to advance graph encoder for stabilization
* `-src_vocab (int)`: Path to source vocabulary

### Acknowledgement

This gated graph neural network is leveraged from https://github.com/JamesChuanggg/ggnn.pytorch.git which is based on the paper [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) by Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel.
