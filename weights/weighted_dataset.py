import codecs
from collections import Counter, defaultdict

import torch
import torchtext.data
import torchtext.vocab
from onmt.IO import PAD_WORD, BOS_WORD, EOS_WORD, __getstate__, __setstate__, \
    join_dicts, ONMTDataset

torchtext.vocab.Vocab.__getstate__ = __getstate__
torchtext.vocab.Vocab.__setstate__ = __setstate__


# TODO: inherit from ONMTDataset adding the target "weights"

class ONMTWeightedDataset(ONMTDataset):
    """Defines a dataset for machine translation with the choice of weighting
    the targets (or source-target pairs) for a measure of 'trustworthiness'."""

    def __init__(self, src_path, tgt_path, fields, opt,
                 src_img_dir=None, dw_path=None, **kwargs):
        """
        Create a TranslationDataset given paths and fields.

        src_path: location of source-side data
        tgt_path: location of target-side data or None. If it exists, it
                  source and target data must be the same length.
        fields:
        src_img_dir: if not None, uses images instead of text for the
                     source. TODO: finish
        """
        if src_img_dir:
            self.type_ = "img"
        else:
            self.type_ = "text"

        if self.type_ == "text":
            self.src_vocabs = []
            src_truncate = 0 if opt is None else opt.src_seq_length_trunc
            src_data = self._read_corpus_file(src_path, src_truncate)
            src_examples = self._construct_examples(src_data, "src")
            self.nfeatures = src_data[0][2]
        else:
            # TODO finish this.
            if not transforms:
                load_image_libs()

        if tgt_path:
            tgt_truncate = 0 if opt is None else opt.tgt_seq_length_trunc
            tgt_data = self._read_corpus_file(tgt_path, tgt_truncate)
            assert len(src_data) == len(tgt_data), \
                "Len src and tgt do not match"
            tgt_examples = self._construct_examples(tgt_data, "tgt")
        else:
            tgt_examples = None

        # datum-weights
        if tgt_path and dw_path:
            # dw_examples = [{"dw": 1} for _ in tgt_data]
            # dw_truncate = 0
            # dw_data = self._read_corpus_file(dw_path, dw_truncate)
            with codecs.open(dw_path, "r", "utf-8") as corpus_file:
                # lines = (line.split() for line in corpus_file)
                dw_data = [float(line) for line in corpus_file]

            assert len(src_data) == len(dw_data), \
                "Len src and dw do not match"
            dw_examples = [{"dw": dw} for dw in dw_data]
        else:
            dw_examples = None

        # examples: one for each src line or (src, tgt) line pair.
        # Each element is a dictionary whose keys represent at minimum
        # the src tokens and their indices and potentially also the
        # src and tgt features and alignment information.
        if tgt_examples and dw_path:
            examples = [join_dicts(src, tgt, dw)
                        for src, tgt, dw in zip(src_examples,
                                                tgt_examples,
                                                dw_examples)]
        elif tgt_examples:
            examples = [join_dicts(src, tgt)
                        for src, tgt in zip(src_examples, tgt_examples)]
        else:
            examples = src_examples
        for i, example in enumerate(examples):
            example["indices"] = i

        if opt is None or opt.dynamic_dict:
            for example in examples:
                src = example["src"]
                src_vocab = torchtext.vocab.Vocab(Counter(src))
                self.src_vocabs.append(src_vocab)
                # mapping source tokens to indices in the dynamic dict
                src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])

                self.src_vocabs.append(src_vocab)
                example["src_map"] = src_map

                if "tgt" in example:
                    tgt = example["tgt"]
                    mask = torch.LongTensor(
                        [0] + [src_vocab.stoi[w] for w in tgt] + [0])
                    example["alignment"] = mask

        keys = examples[0].keys()
        fields = [(k, fields[k]) for k in keys]
        examples = [torchtext.data.Example.fromlist([ex[k] for k in keys],
                                                    fields)
                    for ex in examples]

        def filter_pred(example):
            return 0 < len(example.src) <= opt.src_seq_length \
                and 0 < len(example.tgt) <= opt.tgt_seq_length

        super(ONMTDataset, self).__init__(examples, fields,
                                          filter_pred if opt is not None
                                          else None)

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

        # Added datum weight field
        fields["dw"] = torchtext.data.Field()

        def make_src(data, _):
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(src_size, len(data), src_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=make_src, sequential=False)

        def make_tgt(data, _):
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[:sent.size(0), i] = sent
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            postprocessing=make_tgt, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            sequential=False)

        return fields

    @staticmethod
    def load_fields(vocab):
        vocab = dict(vocab)
        fields = ONMTWeightedDataset.get_fields(
            len(ONMTWeightedDataset.collect_features(vocab)))
        for k, v in vocab.items():
            # Hack. Can't pickle defaultdict :(
            v.stoi = defaultdict(lambda: 0, v.stoi)
            fields[k].vocab = v
        return fields


def load_image_libs():
    "Conditional import of torch image libs."
    global Image, transforms
    from PIL import Image
    from torchvision import transforms
