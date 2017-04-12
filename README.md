# OpenNMT: Open-Source Neural Machine Translation

This is a [Pytorch](https://github.com/pytorch/pytorch)
port of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system.

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>

## Quickstart

## Some useful tools:

The example below uses the Moses tokenizer (http://www.statmt.org/moses/) to prepare the data and the moses BLEU script for evaluation.

```bash
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
```

## WMT'16 Multimodal Translation: Multi30k (de-en)

An example of training for the WMT'16 Multimodal Translation task (http://www.statmt.org/wmt16/multimodal-task.html).

### 0) Download the data.

```bash
mkdir -p data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
wget https://staff.fnwi.uva.nl/d.elliott/wmt16/mmt16_task1_test.tgz && tar -xf mmt16_task1_test.tgz -C data/multi30k && rm mmt16_task1_test.tgz
```

### 1) Preprocess the data.

```bash
for l in en de; do for f in data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
for l in en de; do for f in data/multi30k/*.$l; do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
python preprocess.py -train_src data/multi30k/train.en.atok -train_tgt data/multi30k/train.de.atok -valid_src data/multi30k/val.en.atok -valid_tgt data/multi30k/val.de.atok -save_data data/multi30k.atok.low -lower
```

### 2) Train the model.

```bash
python train.py -data data/multi30k.atok.low.train.pt -save_model multi30k_model -gpus 0
```

### 3) Translate sentences.

```bash
python translate.py -gpu 0 -model multi30k_model_e13_*.pt -src data/multi30k/test.en.atok -tgt data/multi30k/test.de.atok -replace_unk -verbose -output multi30k.test.pred.atok
```

### 4) Evaluate.

```bash
perl multi-bleu.perl data/multi30k/test.de.atok < multi30k.test.pred.atok
```

## Pretrained Models

The following pretrained models can be downloaded and used with translate.py (These were trained with an older version of the code; they will be updated soon).

- [onmt_model_en_de_200k](https://s3.amazonaws.com/pytorch/examples/opennmt/models/onmt_model_en_de_200k-4783d9c3.pt): An English-German translation model based on the 200k sentence dataset at [OpenNMT/IntegrationTesting](https://github.com/OpenNMT/IntegrationTesting/tree/master/data). Perplexity: 21.
- [onmt_model_en_fr_b1M](https://s3.amazonaws.com/pytorch/examples/opennmt/models/onmt_model_en_fr_b1M-261c69a7.pt): An English-French model trained on benchmark-1M. Perplexity: 4.85.

## Release Notes

The following OpenNMT features are implemented:

- multi-layer bidirectional RNNs with attention and dropout
- data preprocessing
- saving and loading from checkpoints
- inference (translation) with batching and beam search

Not yet implemented:

- word features
- multi-GPU
- residual connections

