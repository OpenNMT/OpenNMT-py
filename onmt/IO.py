# -*- coding: utf-8 -*-
import torch
import codecs
import torchtext.data

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


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


class OrderedIterator(torchtext.data.Iterator):
    def create_batches(self):
        self.batches = torchtext.data.pool(self.data(), self.batch_size,
                                           self.sort_key, self.batch_size_fn,
                                           random_shuffler=lambda a: a)


class ONMTDataset(torchtext.data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        "Sort in reverse size order"
        return -len(ex.src)

    def __init__(self, src_path, tgt_path, fields, opt,
                 src_img_dir=None, **kwargs):
        "Create a TranslationDataset given paths and fields."
        if src_img_dir:
            self.type_ = "img"
        else:
            self.type_ = "text"

        examples = []
        src_words = []
        with codecs.open(src_path, "r", "utf-8") as src_file:
            for i, src_line in enumerate(src_file):
                src_line = src_line.split()

                if self.type_ == "text":
                    # Check truncation condition.
                    if opt is not None and opt.src_seq_length_trunc != 0:
                        src_line = src_line[:opt.src_seq_length_trunc]
                    src, src_feats, _ = extractFeatures(src_line)
                    d = {"src": src}
                    self.nfeatures = len(src_feats)
                    for j, v in enumerate(src_feats):
                        examples["src_feats_"+str(j)] = v
                    examples.append(d)
                    src_words.append(src)
                else:
                    # TODO finish this.
                    if not transforms:
                        loadImageLibs()
                    # src_data = transforms.ToTensor()(
                    #     Image.open(src_img_dir + "/" + src_line[0]))

        if tgt_path is not None:
            with codecs.open(tgt_path, "r", "utf-8") as tgt_file:
                for i, tgt_line in enumerate(tgt_file):
                    tgt_line = tgt_line.split()

                    # Check truncation condition.
                    if opt is not None and opt.tgt_seq_length_trunc != 0:
                        tgt_line = tgt_line[:opt.tgt_seq_length_trunc]

                    tgt, _, _ = extractFeatures(tgt_line)
                    examples[i]["tgt"] = tgt

                    # Create Alignment
                    mask = torch.ByteTensor(len(src_words[i]),
                                            len(tgt)).fill_(0)
                    for k in range(len(src_words[i])):
                        for j in range(len(tgt)):
                            if src_words[i][k] == tgt[j]:
                                mask[k][j] = 1
                    examples[i]["alignment"] = mask

        keys = examples[0].keys()
        fields = [(k, fields[k]) for k in keys]
        examples = list([torchtext.data.Example.fromlist([ex[k] for k in keys],
                                                         fields)
                         for ex in examples])

        def filter_pred(example):
            return len(example.src) <= opt.src_seq_length \
                and len(example.tgt) <= opt.tgt_seq_length

        super(ONMTDataset, self).__init__(examples, fields,
                                          filter_pred if opt is not None
                                          else None)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def get_fields(src_path=None, tgt_path=None):
        fields = {}
        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)

        # fields = [("src_img", torchtext.data.Field(
        #     include_lengths=True))]

        if src_path is not None:
            with codecs.open(src_path, "r", "utf-8") as src_file:
                src_line = src_file.readline().strip().split()
                _, _, nFeatures = extractFeatures(src_line)
                for j, v in range(nFeatures):
                    fields["src_feats_"+str(j)] = \
                        torchtext.data.Field(pad_token=PAD_WORD)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        def make_alignment(data):
            src_len = max([t.size(0) for t in data])
            tgt_len = max([t.size(1) for t in data])+1
            alignment = torch.ByteTensor(tgt_len, len(data), src_len).fill_(0)
            for i in range(len(data)):
                alignment[1:data[i].size(1)+1, i, :data[i].size(0)] \
                    = data[i].t()
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, tensor_type=make_alignment,
            sequential=False)
        return fields

    @staticmethod
    def build_vocab(train, opt):
        fields = train.fields
        fields["src"].build_vocab(train, max_size=opt.src_vocab_size)
        for j in range(len(fields) - 3):
            fields["src_feat_" + str(j)].build_vocab(train)
        fields["tgt"].build_vocab(train, max_size=opt.tgt_vocab_size)


def loadImageLibs():
    "Conditional import of torch image libs."
    global Image, transforms
    from PIL import Image
    from torchvision import transforms
