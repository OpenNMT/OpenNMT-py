from onmt.constants import DefaultTokens, ModelTask
import pyonmttok
from collections import Counter


class Example(object):
    """Container for each of the parallel corpus examples"""

    def __init__(self, src, src_original, src_feats=None,
                 tgt=None, tgt_original=None, tgt_feats=None,
                 align=None):
        # 'src_original' and 'tgt_original' store the
        # original line before tokenization. These
        # fields are used later on in the feature
        # transforms.
        self.src = src
        self.src_ids = None
        self.src_original = src_original
        self.src_feats = src_feats
        self.src_feats_ids = None

        self.tgt = tgt
        self.tgt_ids = None
        self.tgt_original = tgt_original
        self.tgt_feats = tgt_feats
        self.tgt_feats_ids = None

        # Alignments
        self.align = align

        # Copy mechanism
        self.src_map = None
        self.src_ex_vocab = None
        self.alignment = None

    def tokenize(self):
        self.src = self.src.strip('\n').split()
        self.src_original = \
            self.src_original.strip("\n").split()
        if self.src_feats is not None:
            self.src_feats = \
                [feat.split() for feat in self.src_feats]
        if self.tgt is not None:
            self.tgt = self.tgt.strip('\n').split()
            self.tgt_original = \
                self.tgt_original.strip("\n").split()
        if self.tgt_feats is not None:
            self.tgt_feats = \
                [feat.split() for feat in self.tgt_feats]
        if self.align is not None:
            self.align = self.align.strip('\n').split()

    def add_index(self, idx):
        self.index = idx

    def is_empty(self):
        if len(self.src) == 0:
            return True
        if self.tgt is not None:
            if len(self.tgt) == 0:
                return True
        if self.align is not None:
            if len(self.align) == 0:
                return True
        return False

    def clean(self):
        self.src = ' '.join(self.src)
        if self.src_feats is not None:
            self.src_feats = [' '.join(x) for x in self.src_feats]
        if self.tgt is not None:
            self.tgt = ' '.join(self.tgt)
        if self.tgt_feats is not None:
            self.tgt_feats = [' '.join(x) for x in self.tgt_feats]
        if self.align is not None:
            self.align = ' '.join(self.align)

    def numericalize(self, vocabs):
        data_task = vocabs['data_task']
        assert data_task in [ModelTask.SEQ2SEQ, ModelTask.LANGUAGE_MODEL], \
            f"Something went wrong with task {vocabs['data_task']}"

        src_toks = self.src.split()
        if data_task == ModelTask.SEQ2SEQ:
            self.src_ids = vocabs['src'](src_toks)
        elif data_task == ModelTask.LANGUAGE_MODEL:
            self.src_ids = vocabs['src']([DefaultTokens.BOS] + src_toks)

        if self.src_feats is not None:
            self.src_feats_ids = []
            for fv, feat in zip(vocabs['src_feats'], self.src_feats):
                feat_toks = feat.split()
                if data_task == ModelTask.SEQ2SEQ:
                    self.src_feats_ids.append(fv(feat_toks))
                else:
                    self.src_feats_ids.append(
                        fv([DefaultTokens.BOS] + feat_toks))

        if self.tgt is not None:
            tgt_toks = self.tgt.split()
            if data_task == ModelTask.SEQ2SEQ:
                self.tgt_ids = vocabs['tgt']([DefaultTokens.BOS]
                                             + tgt_toks
                                             + [DefaultTokens.EOS])
            elif data_task == ModelTask.LANGUAGE_MODEL:
                self.tgt_ids = vocabs['tgt'](tgt_toks + [DefaultTokens.EOS])

        if self.tgt_feats is not None:
            self.tgt_feats_ids = []
            for fv, feat in zip(vocabs['tgt_feats'], self.tgt_feats):
                feat_toks = feat.split()
                if data_task == ModelTask.SEQ2SEQ:
                    self.tgt_feats_ids.append(
                        fv([DefaultTokens.BOS] + feat_toks
                            + [DefaultTokens.EOS]))
                else:
                    self.tgt_feats_ids.append(
                        fv(feat_toks + [DefaultTokens.EOS]))

    def addcopykeys(self, vocabs):
        """Create copy-vocab and numericalize with it.
        In-place adds ``"src_map"`` to ``example``. That is the copy-vocab
        numericalization of the tokenized ``example["src"]``. If ``example``
        has a ``"tgt"`` key, adds ``"alignment"`` to example. That is the
        copy-vocab numericalization of the tokenized ``example["tgt"]``. The
        alignment has an initial and final UNK token to match the BOS and EOS
        tokens.
        Args:
            vocabs
            example (dict): An example dictionary with a ``"src"`` key and
                maybe a ``"tgt"`` key. (This argument changes in place!)
        Returns:
            ``example``, changed as described.
        """
        src = self.src.split()
        src_ex_vocab = pyonmttok.build_vocab_from_tokens(
            Counter(src),
            maximum_size=0,
            minimum_frequency=1,
            special_tokens=[DefaultTokens.UNK,
                            DefaultTokens.PAD,
                            DefaultTokens.BOS,
                            DefaultTokens.EOS])
        src_ex_vocab.default_id = src_ex_vocab[DefaultTokens.UNK]
        # make a small vocab containing just the tokens in the source sequence

        # Map source tokens to indices in the dynamic dict.
        self.src_map = src_ex_vocab(src)
        self.src_ex_vocab = src_ex_vocab

        if self.tgt is not None:
            if vocabs['data_task'] == ModelTask.SEQ2SEQ:
                tgt = [DefaultTokens.UNK] + self.tgt.split() \
                        + [DefaultTokens.UNK]
            elif vocabs['data_task'] == ModelTask.LANGUAGE_MODEL:
                tgt = self.tgt.split() + [DefaultTokens.UNK]
            self.alignment = src_ex_vocab(tgt)
