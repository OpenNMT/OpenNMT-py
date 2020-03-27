# Gated Graph Sequence Neural Networks

Graph-to-sequence networks allow information represtable as a graph (such as an annotated NLP sentence or computer code structure as an AST) to be connected to a sequence generator to produce output which can benefit from the graph structure of the input.

The training option `-encoder_type ggnn` implements a GGNN (Gated Graph Neural Network) based on github.com/JamesChuanggg/ggnn.pytorch.git which is based on the paper "Gated Graph Sequence Neural Networks" by Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel.

The ggnn encoder is used for program equivalence proof generation in the paper <a href="https://arxiv.org/abs/2002.06799">Equivalence of Dataflow Graphs via Rewrite Rules Using a Graph-to-Sequence Neural Model</a>. That paper shows the benefit of the graph-to-sequence model over a sequence-to-sequence model for this problem which can be well represented with graphical input. The integration of the ggnn network into the <a href="https://github.com/OpenNMT/OpenNMT-py/">OpenNMT-py</a> system supports attention on the nodes as well as a copy mechanism.

### Dependencies

* There are no additional dependencies beyond the rnn-to-rnn sequeence2sequence requirements.

### Quick Start

To get started, we provide a toy graph-to-sequence example. We assume that the working directory is `OpenNMT-py` throughout this document.

0) Download the data to a sibling directory.

```
cd ..
git clone https://github.com/SteveKommrusch/OpenNMT-py-ggnn-example
source OpenNMT-py-ggnn-example/env.sh
cd OpenNMT-py
```


1) Preprocess the data.

```
python preprocess.py -train_src $data_path/src-train.txt -train_tgt $data_path/tgt-train.txt -valid_src $data_path/src-val.txt -valid_tgt $data_path/tgt-val.txt -src_seq_length 1000 -tgt_seq_length 30 -src_vocab $data_path/srcvocab.txt -tgt_vocab $data_path/tgtvocab.txt -dynamic_dict -save_data $data_path/final 2>&1 > $data_path/preprocess.out
```

2) Train the model.

```
python train.py -data $data_path/final -encoder_type ggnn -layers 2 -decoder_type rnn -rnn_size 256 -learning_rate 0.1 -start_decay_steps 5000 -learning_rate_decay 0.8 -global_attention general -batch_size 32 -word_vec_size 256 -bridge -train_steps 10000 -gpu_ranks 0 -save_checkpoint_steps 5000 -save_model $data_path/final-model -src_vocab $data_path/srcvocab.txt -n_edge_types 9 -state_dim 256 -n_steps 10 -n_node 64 > $data_path/train.final.out
```

3) Translate the graph of 2 equivalent linear algebra expressions into the axiom list which proves them equivalent.

```
python translate.py -model $data_path/final-model_step_10000.pt -src $data_path/src-test.txt -beam_size 5 -n_best 5 -gpu 0 -output $data_path/pred-test_beam5.txt -dynamic_dict 2>&1 > $data_path/translate5.out
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
`sentence tokens` of which are provided before the first `<EOT>`. After
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


### Options

* `-rnn_type (str)`: style of recurrent unit to use, one of [LSTM]
* `-state_dim (int)`: Number of state dimensions in nodes
* `-n_edge_types (int)`: Number of edge types
* `-bidir_edges (bool)`: True if reverse edges should be automatically created
* `-n_node (int)`: Max nodes in graph
* `-bridge_extra_node (bool)`: True indicates only the vector from the 1st extra node (after token listing) should be used for decoder initialization; False indicates all node vectors should be averaged together for decoder initialization
* `-n_steps (int)`: Steps to advance graph encoder for stabilization
* `-src_vocab (int)`: Path to source vocabulary

### Acknowledgement

This gated graph neural network is leveraged from github.com/JamesChuanggg/ggnn.pytorch.git which is based on the paper "Gated Graph Sequence Neural Networks" by Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel.
