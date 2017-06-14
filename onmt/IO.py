import onmt.Constants.UNK_WORD


def loadImageLibs():
    "Conditional import of torch image libs."
    global Image, transforms
    from PIL import Image
    from torchvision import transforms


def readSrcLine(src_line, src_dict, src_feature_dicts, _type="text"):
    srcFeat = None
    if self._type == "text":
        srcWords, srcFeatures, _ = extractFeatures(src_line)
        srcData = src_dict.convertToIdx(srcWords,
                                        onmt.Constants.UNK_WORD)
        if src_feature_dicts:
            srcFeat = [src_feature_dicts[j].
                       convertToIdx(srcFeatures[j],
                                    onmt.Constants.UNK_WORD)
                       for j in range(len(src_feature_dicts))]
    elif _type == "img":
        if not transforms:
            loadImageLibs()
        srcData = [transforms.ToTensor()(
            Image.open(self.opt.src_img_dir + "/" + b[0]))
                   for b in srcBatch]

    return srcData, srcFeat

def readTgtLine(tgt_line, tgt_dict, tgt_feature_dicts, _type="text"):
    tgtWords, tgtFeatures, _ = extractFeatures(tgt_line)
    tgtData = tgt_dict.convertToIdx(tgtWords,
                                    onmt.Constants.UNK_WORD,
                                    onmt.Constants.BOS_WORD,
                                    onmt.Constants.EOS_WORD)]
    if tgtFeatureDicts:
        tgtFeats = [tgt_feature_dicts[j].
                    convertToIdx(tgtFeatures[j],
                                 onmt.Constants.UNK_WORD)
                    for j in range(len(tgt_feature_dicts))]
        
    return tgtData, tgtFeat
            
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
