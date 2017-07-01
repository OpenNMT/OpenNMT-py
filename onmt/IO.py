# -*- coding: utf-8 -*-
import torch
import onmt.Constants


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
