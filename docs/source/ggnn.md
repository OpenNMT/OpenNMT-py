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
python preprocess.py -train_src $data_path/src-train.txt -train_tgt $data_path/tgt-train.txt -valid_src $data_path/src-val.txt -valid_tgt $data_path/tgt-val.txt -src_seq_length 1000 -tgt_seq_length 30 -src_vocab $data_path/vocab.txt -tgt_vocab $data_path/tgtvocab.txt -dynamic_dict -save_data $data_path/final 2>&1 > $data_path/preprocess.out
```

2) Train the model.

```
python train.py -data $data_path/final -encoder_type ggnn -layers 2 -decoder_type rnn -rnn_size 256 -learning_rate 0.1 -start_decay_steps 5000 -learning_rate_decay 0.8 -global_attention general -batch_size 32 -word_vec_size 256 -bridge -train_steps 10000 -gpu_ranks 0 -save_checkpoint_steps 5000 -save_model $data_path/final-model -src_vocab $data_path/vocab.txt -n_edge_types 9 -state_dim 256 -n_steps 10 -n_node 64 > $data_path/train.final.out
```

3) Translate the graph of 2 equivalent linear algebra expressions into the axiom list which proves them equivalent.

```
python translate.py -model $data_path/final-model_step_100000.pt -src $data_path/src-test.txt -beam_size 5 -n_best 5 -gpu 0 -output $data_path/pred-test_beam5.txt -dynamic_dict 2>&1 > $data_path/translate5.out
```

### Graph data format

FIXME: Add description, (use first line in src-test.txt file). Describe standard word tokens, node features, edge format.

### Options

* `-rnn_type (str)`: style of recurrent unit to use, one of [LSTM]
* `-state_dim (int)`: Number of state dimensions in nodes
* `-n_edge_types (int)`: Number of edge types
* `-bidir_edges (bool)`: True if reverse edges should be autocreated
* `-n_node (int)`: Max nodes in graph
* `-bridge_extra_node (bool)`: True indicates only 1st extra node (after token listing) should be used for decoder init.
* `-n_steps (int)`: Steps to advance graph encoder for stabilization
* `-src_vocab (int)`: Path to source vocabulary

### Acknowledgement

This gated graph neural network is leveraged from github.com/JamesChuanggg/ggnn.pytorch.git which is based on the paper "Gated Graph Sequence Neural Networks" by Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel.
