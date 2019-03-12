This is a fork of OpenNMT-py (https://github.com/OpenNMT/OpenNMT-py), v0.4 with modifications to run experiments on AMR-to-text generation discussed in our paper (LINK).

## Install

Follow instructions on [README_OpenNMT-py.md](README_OpenNMT-py.md) to install

## Data

Use https://github.com/sinantie/NeuralAmr to generate the linearized and anonymized data.

## Experiments

Follow these instructions to replicate the experiments reported in Table 1 of the paper.

For each experiment, run the preprocessing script, training script and testing script. For sequential and tree encoders, the preprocessing script to use is ```preproc_amr.sh``` and the evaluation script is ```predict.sh```. For graph encoders, use ```preproc_amr_reent.sh``` and ```predict_reent.sh```. In the following we report the training scripts to use for each experiment. Refer to the paper for the explanation of each model.

### Sequential encoders

Seq: ```train_amr_seq.sh```

### Tree encoders

SeqTreeLSTM: ```train_amr_tree_seq_treelstm.sh```

TreeLSTMSeq: ```train_amr_tree_treelstm_seq.sh```

TreeLSTM: ```train_amr_tree_treelstm.sh```

SeqGCN: ```train_amr_tree_seq_gcn.sh```

GCNSeq: ```train_amr_tree_gcn_seq.sh```

GCN: ```train_amr_tree_gcn.sh```

### Graph encoders encoders

SeqGCN: ```train_amr_graph_seq_gcn.sh```

GCNSeq: ```train_amr_graph_gcn_seq.sh```

GCN: ```train_amr_graph_gcn.sh```

### Evaluation

Use ```recomputeMetrics.sh``` in https://github.com/sinantie/NeuralAmr to evaluate the models.

## Contrastive examples

See ```contrastive_examples/```

## Citation

```
@inproceedings{damonte2019gen,
  title={Structural Neural Encoders for AMR-to-text Generation},
  author={Damonte, Marco and Cohen, Shay B},
  booktitle={Proceedings of NAACL},
  year={2019}
}
```
