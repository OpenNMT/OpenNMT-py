import onmt
from onmt.constants import DefaultTokens
from onmt.modules.sparse_losses import SparsemaxLoss
import torch.nn as nn


class Criterions:

    def __init__(self, opt, vocabs):
        tgt_vocab = vocabs['tgt']
        padding_idx = tgt_vocab[DefaultTokens.PAD]
        unk_idx = tgt_vocab[DefaultTokens.UNK]

        if opt.copy_attn:
            self.tgt_criterion = onmt.modules.CopyGeneratorLoss(
                len(tgt_vocab), opt.copy_attn_force,
                unk_index=unk_idx, ignore_index=padding_idx
            )
        else:
            if opt.generator_function == 'sparsemax':
                self.tgt_criterion = SparsemaxLoss(
                    ignore_index=padding_idx,
                    reduction='sum')
            else:
                self.tgt_criterion = nn.CrossEntropyLoss(
                    ignore_index=padding_idx,
                    reduction='sum',
                    label_smoothing=opt.label_smoothing)

        # Add as many criterios as tgt features we have
        self.feats_criterions = []
        if 'tgt_feats' in vocabs:
            for feat_vocab in vocabs["tgt_feats"]:
                padding_idx = feat_vocab[DefaultTokens.PAD]
                self.feats_criterions.append(
                    nn.CrossEntropyLoss(
                        ignore_index=padding_idx,
                        reduction='sum')
                )