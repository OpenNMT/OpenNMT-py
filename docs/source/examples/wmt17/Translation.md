# Translation WMT17 en-de

## Dependencies

### PyTorch

<https://pytorch.org/get-started/locally/>

```bash
pip3 install torch torchvision torchaudio
```

### Apex

This is highly recommended to have fast performance.

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
cd ..
```

### Subword-NMT

```bash
pip3 install subword-nmt
```

### OpenNMT-py

<https://github.com/OpenNMT/OpenNMT-py>

```bash
git clone https://github.com/OpenNMT/OpenNMT-py.git
cd OpenNMT-py
pip3 install --editable ./
```

## Running WMT17 EN-DE

### Get Data and prepare

WMT17 English-German data set:

```bash
cd docs/source/examples
bash wmt17/prepare_wmt_ende_data.sh
```

### Train

Training the following big transformer for 50K steps takes less than 10 hours on a single RTX 4090

```bash
python3 ../../../onmt/bin/build_vocab.py --config wmt17/wmt17_ende.yaml --n_sample -1
python3 ../../../onmt/bin/train.py --config wmt17/wmt17_ende.yaml
```

Translate test sets with various settings on local GPU and CPUs.

```bash
python3 ../../../onmt/bin/translate.py --src wmt17_en_de/test.src.bpe --model wmt17_en_de/bigwmt17_step_50000.pt --beam_size 5 --batch_size 4096 --batch_type tokens --output wmt17_en_de/pred.trg.bpe --gpu 0
sed -re 's/@@( |$)//g' < wmt17_en_de/pred.trg.bpe > wmt17_en_de/pred.trg.tok
sacrebleu -tok none wmt17_en_de/test.trg < wmt17_en_de/pred.trg.tok
```

BLEU scored at 40K, 45K, 50K steps on the test set (Newstest2016)

```
{
 "name": "BLEU",
 "score": 35.4,
 "signature": "nrefs:1|case:mixed|eff:no|tok:none|smooth:exp|version:2.0.0",
 "verbose_score": "66.2/41.3/28.5/20.3 (BP = 0.998 ratio = 0.998 hyp_len = 64244 ref_len = 64379)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "none",
 "smooth": "exp",
 "version": "2.0.0"
}
{
 "name": "BLEU",
 "score": 35.2,
 "signature": "nrefs:1|case:mixed|eff:no|tok:none|smooth:exp|version:2.0.0",
 "verbose_score": "65.9/41.0/28.3/20.2 (BP = 1.000 ratio = 1.000 hyp_len = 64357 ref_len = 64379)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "none",
 "smooth": "exp",
 "version": "2.0.0"
}
{
 "name": "BLEU",
 "score": 35.1,
 "signature": "nrefs:1|case:mixed|eff:no|tok:none|smooth:exp|version:2.0.0",
 "verbose_score": "66.2/41.2/28.4/20.3 (BP = 0.992 ratio = 0.992 hyp_len = 63885 ref_len = 64379)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "none",
 "smooth": "exp",
 "version": "2.0.0"
}

```
