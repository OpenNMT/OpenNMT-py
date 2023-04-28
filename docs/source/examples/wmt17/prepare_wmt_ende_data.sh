#!/usr/bin/env bash

if ! command -v subword-nmt &>/dev/null; then
  echo "Please install Subword NMT: pip3 install subword-nmt"
  exit 2
fi

mkdir -p wmt17_en_de
cd wmt17_en_de
if true; then
wget 'http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.en.gz'
wget 'http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.de.gz'
wget 'http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/dev.tgz'
tar xf dev.tgz

ln -s corpus.tc.en.gz train.src.gz
ln -s corpus.tc.de.gz train.trg.gz
cat newstest2014.tc.en newstest2015.tc.en >dev.src
cat newstest2014.tc.de newstest2015.tc.de >dev.trg
ln -s newstest2016.tc.en test.src
ln -s newstest2016.tc.de test.trg

zcat train.src.gz train.trg.gz |subword-nmt learn-bpe -s 32000 >codes
for LANG in src trg; do
  zcat train.$LANG.gz |subword-nmt apply-bpe -c codes >train.$LANG.bpe
  for SET in dev test; do
    subword-nmt apply-bpe -c codes <$SET.$LANG >$SET.$LANG.bpe
  done
done
fi
python3 ../$(dirname $0)/filter_train.py
paste -d '\t' train.src.bpe.filter train.trg.bpe.filter | shuf | awk -v FS="\t" '{ print $1 > "train.src.bpe.shuf" ; print $2 > "train.trg.bpe.shuf" }'
