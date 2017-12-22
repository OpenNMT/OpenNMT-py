from __future__ import division, unicode_literals

import torch
import onmt.io


class TranslationBuilder(object):
    def __init__(self, data, fields, n_best, replace_unk, has_tgt):
        self.data = data
        self.fields = fields
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.has_tgt = has_tgt

    def _build_target_tokens(self, src, src_vocab, src_raw, pred, attn):
        vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                tokens.append(src_vocab.itos[tok - len(vocab)])
            if tokens[-1] == onmt.io.EOS_WORD:
                tokens = tokens[:-1]
                break
        if self.replace_unk and (attn is not None) and (src is not None):
            for i in range(len(tokens)):
                if tokens[i] == vocab.itos[onmt.io.UNK]:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src_raw[maxIndex[0]]
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, predScore, attn, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        # Sorting
        inds, perm = torch.sort(batch.indices.data)
        data_type = self.data.data_type
        if data_type == 'text':
            src = batch.src[0].data.index_select(1, perm)
        else:
            src = None

        if self.has_tgt:
            tgt = batch.tgt.data.index_select(1, perm)
        else:
            tgt = None

        translations = []
        for b in range(batch_size):
            if data_type == 'text':
                src_vocab = self.data.src_vocabs[inds[b]]
                src_raw = self.data.examples[inds[b]].src
            else:
                src_vocab = None
                src_raw = None
            pred_sents = [self._build_target_tokens(
                src[:, b] if src is not None else None,
                src_vocab, src_raw,
                preds[b][n], attn[b][n])
                          for n in range(self.n_best)]
            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    src[:, b] if src is not None else None,
                    src_vocab, src_raw,
                    tgt[1:, b] if tgt is not None else None, None)

            translation = Translation(src[:, b], src_raw, pred_sents,
                                      attn[b], predScore[b], gold_sent,
                                      gold_score[b])
            translations.append(translation)

        return translations


class Translation(object):
    def __init__(self, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        output = '\nSENT {}: {}\n'.format(sent_number, ' '.join(self.src_raw))

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        print("PRED SCORE: {:.4f}".format(best_score))

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}".format(self.gold_score))

        if len(self.pred_sents) > 1:
            print('\nBEST HYP:')
            for score, sent in zip(self.pred_score, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
