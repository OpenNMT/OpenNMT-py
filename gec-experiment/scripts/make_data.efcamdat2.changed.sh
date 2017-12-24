OPENNMT_HOME=/home/howard/gec/OpenNMT-py
TRAIN_DATA_HOME=/home/howard/gec/dataset/efcamdat
VALID_DATA_HOME=/home/howard/gec/dataset/jfleg

python $OPENNMT_HOME/preprocess.py \
  -train_src $TRAIN_DATA_HOME/efcamdat2.changed.src.txt \
  -train_tgt $TRAIN_DATA_HOME/efcamdat/efcamdat2.changed.tgt.txt \
  -valid_src $VALID_DATA_HOME/dev/dev.src \
  -valid_tgt $VALID_DATA_HOME/dev/dev.ref0 \
  -save_data $OPENNMT_HOME/data/efcamdat2.changed \
  -dynamic_dict -share_vocab
