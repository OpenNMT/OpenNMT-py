#!/usr/bin/env bash

if ! command -v subword-nmt &>/dev/null; then
  echo "Please install Subword NMT: pip3 install subword-nmt"
  exit 2
fi

mkdir -p wmt17_ru_en
cd wmt17_ru_en

wget 'http://data.statmt.org/wmt17/translation-task/preprocessed/ru-en/corpus.tc.ru.gz'
wget 'http://data.statmt.org/wmt17/translation-task/preprocessed/ru-en/corpus.tc.en.gz'
wget 'http://data.statmt.org/wmt17/translation-task/preprocessed/ru-en/dev.tgz'
tar xf dev.tgz

ln -s corpus.tc.ru.gz train.src.gz
ln -s corpus.tc.en.gz train.trg.gz
cat newstest2014.tc.ru newstest2015.tc.ru >dev.src
cat newstest2014.tc.en newstest2015.tc.en >dev.trg
ln -s newstest2016.tc.ru test.src
ln -s newstest2016.tc.en test.trg

zcat train.src.gz train.trg.gz |subword-nmt learn-bpe -s 32000 >codes
for LANG in src trg; do
  zcat train.$LANG.gz |subword-nmt apply-bpe -c codes >train.$LANG.bpe
  for SET in dev test; do
    subword-nmt apply-bpe -c codes <$SET.$LANG >$SET.$LANG.bpe
  done
done

python3 $(dirname $0)/filter_train.py

