# -*- coding: utf-8 -*-
import torch
import codecs
import torchtext.data
import torchtext.vocab
from collections import Counter, defaultdict

PAD_WORD = '<blank>'
UNK = 0
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


def merge_vocabs(vocabs, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = Counter()
    # take the counts of the disjoint union of all the vocabs
    for vocab in vocabs:
        # XXX note that `vocab.freqs` does not contain special symbols
        for word, count in vocab.freqs.most_common():
            if word not in merged:
                merged[word] = 0
            merged[word] += count
    return torchtext.vocab.Vocab(merged,
                                 specials=[PAD_WORD, BOS_WORD, EOS_WORD],
                                 max_size=vocab_size)


def make_features(batch, fields):
    # TODO: This is bit hacky, add to batch somehow.
    f = ONMTDataset.collect_features(fields)
    cat = [batch.src[0]] + [batch.__dict__[k] for k in f]
    cat = [c.unsqueeze(2) for c in cat]
    return torch.cat(cat, 2)


class OrderedIterator(torchtext.data.Iterator):
    def create_batches(self):
        if self.train:
            self.batches = torchtext.data.pool(
                self.data(), self.batch_size,
                self.sort_key, self.batch_size_fn,
                random_shuffler=self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


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
        self.src_vocabs = []
        with codecs.open(src_path, "r", "utf-8") as src_file:
            for i, src_line in enumerate(src_file):
                src_line = src_line.split()
                # if len(src_line) == 0:
                #     skip[i] = True
                #     continue
                if self.type_ == "text":
                    # Check truncation condition.
                    if opt is not None and opt.src_seq_length_trunc != 0:
                        src_line = src_line[:opt.src_seq_length_trunc]
                    src, src_feats, _ = extractFeatures(src_line)
                    d = {"src": src, "indices": i}
                    self.nfeatures = len(src_feats)
                    for j, v in enumerate(src_feats):
                        d["src_feat_"+str(j)] = v
                    examples.append(d)
                    src_words.append(src)

                    # Create dynamic dictionaries
                    if opt is None or opt.dynamic_dict:
                        # a temp vocab of a single source example
                        src_vocab = torchtext.vocab.Vocab(Counter(src))

                        # mapping source tokens to indices in the dynamic dict
                        src_map = torch.LongTensor(len(src)).fill_(0)
                        for j, w in enumerate(src):
                            src_map[j] = src_vocab.stoi[w]

                        self.src_vocabs.append(src_vocab)
                        examples[i]["src_map"] = src_map

                else:
                    # TODO finish this.
                    if not transforms:
                        loadImageLibs()
                    # src_data = transforms.ToTensor()(
                    #     Image.open(src_img_dir + "/" + src_line[0]))

        if tgt_path is not None:
            with codecs.open(tgt_path, "r", "utf-8") as tgt_file:
                for i, tgt_line in enumerate(tgt_file):
                    # if i in skip:
                    #     continue
                    tgt_line = tgt_line.split()

                    # Check truncation condition.
                    if opt is not None and opt.tgt_seq_length_trunc != 0:
                        tgt_line = tgt_line[:opt.tgt_seq_length_trunc]

                    tgt, _, _ = extractFeatures(tgt_line)
                    examples[i]["tgt"] = tgt

                    if opt is None or opt.dynamic_dict:
                        src_vocab = self.src_vocabs[i]
                        # Map target tokens to indices in the dynamic dict
                        mask = torch.LongTensor(len(tgt)+2).fill_(0)
                        for j in range(len(tgt)):
                            mask[j+1] = src_vocab.stoi[tgt[j]]
                        examples[i]["alignment"] = mask
                assert i + 1 == len(examples), "Len src and tgt do not match"
        keys = examples[0].keys()
        fields = [(k, fields[k]) for k in keys]
        examples = list([torchtext.data.Example.fromlist([ex[k] for k in keys],
                                                         fields)
                         for ex in examples])

        def filter_pred(example):
            return 0 < len(example.src) <= opt.src_seq_length \
                and 0 < len(example.tgt) <= opt.tgt_seq_length

        super(ONMTDataset, self).__init__(examples, fields,
                                          filter_pred if opt is not None
                                          else None)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(ONMTDataset, self).__reduce_ex__()

    def collapseCopyScores(self, scores, batch, tgt_vocab):
        """Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            index = batch.indices.data[b]
            src_vocab = self.src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    scores[:, b, ti] += scores[:, b, offset + i]
                    scores[:, b, offset + i].fill_(1e-20)
        return scores

    @staticmethod
    def load_fields(vocab):
        vocab = dict(vocab)
        fields = ONMTDataset.get_fields(
            len(ONMTDataset.collect_features(vocab)))
        for k, v in vocab.items():
            # Hack. Can't pickle defaultdict :(
            v.stoi = defaultdict(lambda: 0, v.stoi)
            fields[k].vocab = v
        return fields

    @staticmethod
    def save_vocab(fields):
        vocab = []
        for k, f in fields.items():
            if 'vocab' in f.__dict__:
                f.vocab.stoi = dict(f.vocab.stoi)
                vocab.append((k, f.vocab))
        return vocab

    @staticmethod
    def collect_features(fields):
        feats = []
        j = 0
        while True:
            key = "src_feat_" + str(j)
            if key not in fields:
                break
            feats.append(key)
            j += 1
        return feats

    @staticmethod
    def collect_feature_dicts(fields):
        feature_dicts = []
        j = 0
        while True:
            key = "src_feat_" + str(j)
            if key not in fields:
                break
            feature_dicts.append(fields[key].vocab)
            j += 1
        return feature_dicts

    @staticmethod
    def get_fields(nFeatures=0):
        fields = {}
        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)

        # fields = [("src_img", torchtext.data.Field(
        #     include_lengths=True))]

        for j in range(nFeatures):
            fields["src_feat_"+str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        def make_src(data, _):
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.FloatTensor(src_size, len(data),
                                          src_vocab_size).fill_(0)
            for i in range(len(data)):
                for j, t in enumerate(data[i]):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=make_src, sequential=False)

        def make_tgt(data, _):
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.LongTensor(tgt_size, len(data)).fill_(0)
            for i in range(len(data)):
                alignment[:data[i].size(0), i] = data[i]
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            postprocessing=make_tgt, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            sequential=False)

        return fields

    @staticmethod
    def build_vocab(train, opt):
        fields = train.fields
        fields["src"].build_vocab(train, max_size=opt.src_vocab_size,
                                  min_freq=opt.src_words_min_frequency)
        for j in range(train.nfeatures):
            fields["src_feat_" + str(j)].build_vocab(train)
        fields["tgt"].build_vocab(train, max_size=opt.tgt_vocab_size,
                                  min_freq=opt.tgt_words_min_frequency)

        # Merge the input and output vocabularies.
        if opt.share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            merged_vocab = merge_vocabs(
                [fields["src"].vocab, fields["tgt"].vocab],
                vocab_size=opt.src_vocab_size)
            fields["src"].vocab = merged_vocab
            fields["tgt"].vocab = merged_vocab


def loadImageLibs():
    "Conditional import of torch image libs."
    global Image, transforms
    from PIL import Image
    from torchvision import transforms
