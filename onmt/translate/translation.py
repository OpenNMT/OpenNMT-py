""" Translation main class """
import os
import torch
from onmt.constants import DefaultTokens
from onmt.utils.alignment import build_align_pharaoh


class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data ():
       vocabs ():
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
    """

    def __init__(self, data, vocabs, n_best=1, replace_unk=False,
                 phrase_table=""):
        self.data = data
        self.vocabs = vocabs
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.phrase_table_dict = {}
        if phrase_table != "" and os.path.exists(phrase_table):
            with open(phrase_table) as phrase_table_fd:
                for line in phrase_table_fd:
                    phrase_src, phrase_trg = line.rstrip("\n").split(
                        DefaultTokens.PHRASE_TABLE_SEPARATOR)
                    self.phrase_table_dict[phrase_src] = phrase_trg

    def _build_target_tokens(self, src, src_raw, pred, attn):
        tokens = []

        for tok in pred:
            if tok < len(self.vocabs['tgt']):
                tokens.append(self.vocabs['tgt'].lookup_index(tok))
            else:
                vl = len(self.vocabs['tgt'])
                tokens.append(self.vocabs['src'].lookup_index(tok - vl))
            if tokens[-1] == DefaultTokens.EOS:
                tokens = tokens[:-1]
                break
        if self.replace_unk and attn is not None and src is not None:
            for i in range(len(tokens)):
                if tokens[i] == DefaultTokens.UNK:
                    _, max_index = attn[i][:len(src_raw)].max(0)
                    tokens[i] = src_raw[max_index.item()]
                    if self.phrase_table_dict:
                        src_tok = src_raw[max_index.item()]
                        if src_tok in self.phrase_table_dict:
                            tokens[i] = self.phrase_table_dict[src_tok]
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = len(batch['srclen'])

        preds, pred_score, attn, align, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["alignment"],
                        translation_batch["gold_score"],
                        batch['indices']),
                    key=lambda x: x[-1])))

        if not any(align):  # when align is a empty nested list
            align = [None] * batch_size

        # Sorting
        inds, perm = torch.sort(batch['indices'])

        src = batch['src'][:, :, 0].index_select(0, perm)
        if 'tgt' in batch.keys():
            tgt = batch['tgt'][:, :, 0].index_select(0, perm)
        else:
            tgt = None

        translations = []

        for b in range(batch_size):
            # src_raw = self.data.examples[inds[b]].src[0]
            src_raw = None
            pred_sents = [self._build_target_tokens(
                src[b, :] if src is not None else None,
                src_raw,
                preds[b][n],
                align[b][n] if align[b] is not None else attn[b][n])
                for n in range(self.n_best)]
            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    src[b, :] if src is not None else None,
                    src_raw,
                    tgt[b, 1:] if tgt is not None else None, None)

            translation = Translation(
                src[b, :] if src is not None else None,
                src_raw, pred_sents, attn[b], pred_score[b],
                gold_sent, gold_score[b], align[b]
            )
            translations.append(translation)

        return translations


class Translation(object):
    """Container for a translated sentence.

    Attributes:
        src (LongTensor): Source word IDs.
        src_raw (List[str]): Raw source words.
        pred_sents (List[List[str]]): Words from the n-best translations.
        pred_scores (List[List[float]]): Log-probs of n-best translations.
        attns (List[FloatTensor]) : Attention distribution for each
            translation.
        gold_sent (List[str]): Words from gold translation.
        gold_score (List[float]): Log-prob of gold translation.
        word_aligns (List[FloatTensor]): Words Alignment distribution for
            each translation.
    """

    __slots__ = ["src", "src_raw", "pred_sents", "attns", "pred_scores",
                 "gold_sent", "gold_score", "word_aligns"]

    def __init__(self, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score, word_aligns):
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score
        self.word_aligns = word_aligns

    def log(self, sent_number):
        """
        Log translation.
        """

        msg = ['\nSENT {}: {}\n'.format(sent_number, self.src_raw)]

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        msg.append('PRED {}: {}\n'.format(sent_number, pred_sent))
        msg.append("PRED SCORE: {:.4f}\n".format(best_score))

        if self.word_aligns is not None:
            pred_align = self.word_aligns[0]
            pred_align_pharaoh = build_align_pharaoh(pred_align)
            pred_align_sent = ' '.join(pred_align_pharaoh)
            msg.append("ALIGN: {}\n".format(pred_align_sent))

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            msg.append('GOLD {}: {}\n'.format(sent_number, tgt_sent))
            msg.append(("GOLD SCORE: {:.4f}\n".format(self.gold_score)))
        if len(self.pred_sents) > 1:
            msg.append('\nBEST HYP:\n')
            for score, sent in zip(self.pred_scores, self.pred_sents):
                msg.append("[{:.4f}] {}\n".format(score, sent))

        return "".join(msg)
