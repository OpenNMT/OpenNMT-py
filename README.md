# PyOpenNMT: Open-Source Neural Machine Translation

This is a [Pytorch](https://github.com/pytorch/pytorch)
port of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system. Full documentation is available [here](http://opennmt.net/OpenNMT-py).

This code is still in heavy development (pre-version 0.1). We recommend forking if you want a stable version. 

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>

## Features

The following OpenNMT features are implemented:

- multi-layer bidirectional RNNs with attention and dropout
- data preprocessing
- saving and loading from checkpoints
- inference (translation) with batching and beam search
- multi-GPU

Beta Features:
- Context gate
- Multiple source and target RNN (lstm/gru) types and attention (dotprod/mlp) types 
- Image-to-text processing
- Source word features
- "Attention is all you need" 
- TensorBoard/Crayon logging
- Copy, coverage, and structured attention


## Quickstart

## Step 1: Preprocess the data

```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

We will be working with some example data in `data/` folder.

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `demo.src.dict`: Dictionary of source vocab to index mappings.
* `demo.tgt.dict`: Dictionary of target vocab to index mappings.
* `demo.train.pt`: serialized PyTorch file containing vocabulary, training and validation data


Internally the system never touches the words themselves, but uses these indices.

## Step 2: Train the model

```bash
python train.py -data data/demo.train.pt -save_model demo-model 
```

The main train command is quite simple. Minimally it takes a data file
and a save file.  This will run the default model, which consists of a
2-layer LSTM with 500 hidden units on both the encoder/decoder. You
can also add `-gpus 1` to use (say) GPU 1.

## Step 3: Translate

```bash
python translate.py -model demo-model_epochX_PPL.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose
```

Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into `pred.txt`.

!!! note "Note"
    The predictions are going to be quite terrible, as the demo dataset is small. Try running on some larger datasets! For example you can download millions of parallel sentences for [translation](http://www.statmt.org/wmt16/translation-task.html) or [summarization](https://github.com/harvardnlp/sent-summary).

## Some useful tools:


## Full Translation Example 

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
# Delete the last line of val and training files.
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


