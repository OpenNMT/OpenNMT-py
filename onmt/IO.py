# -*- coding: utf-8 -*-
import torch
import codecs
import onmt.Constants
import torchtext.data

def extractFeatures(tokens):
    "Given a list of token separate out words and features (if any)."
    words = []
    features = []
    numFeatures = None

    for t in range(len(tokens)):
        field = tokens[t].split(u"￨")
        word = field[0]
        if len(word) > 0:
            words.append(word)

            if numFeatures is None:
                numFeatures = len(field) - 1
            else:
                assert (len(field) - 1 == numFeatures), \
                    "all words must have the same number of features"

            if len(field) > 1:
                for i in range(1, len(field)):
                    if len(features) <= i-1:
                        features.append([])
                    features[i - 1].append(field[i])
                    assert (len(features[i - 1]) == len(words))
    return words, features, numFeatures if numFeatures else 0


class ONMTDataset(torchtext.data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, src_path, tgt_path, fields, opt, **kwargs):
        """Create a TranslationDataset given paths and fields.
        """

        examples = []

        with codecs.open(src_path, "r", "utf-8") as src_file:
            for i, src_line in enumerate(src_file):
                src_line = src_line.split()

                # Check truncation condition.
                if opt.src_seq_length_trunc != 0:
                    src_line = srcLine[:opt.src_seq_length_trunc]
                src, srcFeats, _  = extractFeatures(src_line)
                examples.append([src] + srcFeats)

        with codecs.open(tgt_path, "r", "utf-8") as tgt_file:
            for i, tgt_line in enumerate(tgt_file):
                tgt_line = tgt_line.split()

                # Check truncation condition.
                if opt.tgt_seq_length_trunc != 0:
                    tgt_line = tgt_line[:opt.tgt_seq_length_trunc]

                tgt, _, _  = extractFeatures(tgt_line)
                examples[i].append(tgt)

        examples = list([torchtext.data.Example.fromlist(ex, fields)
                         for ex in examples])

        def filter_pred(example):
            return len(example.src) <= opt.src_seq_length \
                and len(example.tgt) <= opt.tgt_seq_length

        super(ONMTDataset, self).__init__(examples, fields, filter_pred)

    @staticmethod
    def get_fields(src_path, tgt_path):
        fields = [("src", torchtext.data.Field(pad_token=onmt.Constants.PAD_WORD))]
        with codecs.open(src_path, "r", "utf-8") as src_file:
            src_line = src_file.readline().strip().split()
            _, _, nFeatures  = extractFeatures(src_line)
            for j in range(nFeatures):
                fields.append(("src_feat_" + str(j),
                               torchtext.data.Field(pad_token=onmt.Constants.PAD_WORD)))
        fields.append(("tgt", torchtext.data.Field(init_token=onmt.Constants.BOS_WORD,
                                                   eos_token=onmt.Constants.EOS_WORD,
                                                   pad_token=onmt.Constants.PAD_WORD)))
        return fields

    @staticmethod
    def build_vocab(train, fields, opt):
        fields = dict(fields)
        vocabs = []
        vocabs.append(fields["src"] \
                      .build_vocab(train, max_size=opt.src_vocab_size))
        for j in range(len(fields) - 2):
            vocabs.append(fields["src_feat_" + str(j)].build_vocab(self))
        vocabs.append(fields["tgt"] \
                      .build_vocab(train, max_size=opt.tgt_vocab_size))
        return vocabs


def loadImageLibs():
    "Conditional import of torch image libs."
    global Image, transforms
    from PIL import Image
    from torchvision import transforms


def align(src_tokens, tgt_tokens):
    """
    Given two sequences of tokens, return
    a mask of where there is overlap.

    Returns:
        mask: tgt_len x src_len
    """
    mask = torch.ByteTensor(len(src_tokens), len(tgt_tokens)).fill_(0)

    for i in range(len(src_tokens)):
        for j in range(len(tgt_tokens)):
            if src_tokens[i] == tgt_tokens[j]:
                mask[i][j] = 1
    return mask


def readSrcLine(src_line, src_dict, src_feature_dicts,
                _type="text", src_img_dir=""):
    srcFeats = None
    if _type == "text":
        srcWords, srcFeatures, _ = extractFeatures(src_line)
        srcData = src_dict.convertToIdx(srcWords,
                                        onmt.Constants.UNK_WORD)
        if src_feature_dicts:
            srcFeats = [src_feature_dicts[j].
                        convertToIdx(srcFeatures[j],
                                     onmt.Constants.UNK_WORD)
                        for j in range(len(src_feature_dicts))]
    elif _type == "img":
        if not transforms:
            loadImageLibs()
        srcData = transforms.ToTensor()(
            Image.open(src_img_dir + "/" + srcWords[0]))

    return srcWords, srcData, srcFeats


def readTgtLine(tgt_line, tgt_dict, tgt_feature_dicts, _type="text"):
    tgtFeats = None
    tgtWords, tgtFeatures, _ = extractFeatures(tgt_line)
    tgtData = tgt_dict.convertToIdx(tgtWords,
                                    onmt.Constants.UNK_WORD,
                                    onmt.Constants.BOS_WORD,
                                    onmt.Constants.EOS_WORD)
    if tgt_feature_dicts:
        tgtFeats = [tgt_feature_dicts[j].
                    convertToIdx(tgtFeatures[j],
                                 onmt.Constants.UNK_WORD)
                    for j in range(len(tgt_feature_dicts))]

    return tgtWords, tgtData, tgtFeats
