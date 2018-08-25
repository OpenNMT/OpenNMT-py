# coding: utf-8

from collections import Counter
from itertools import chain
import torch
from torchtext.vocab import Vocab
from torchtext.data import Example, Dataset

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def _join_dicts(*args):
    """
    args: dictionaries with disjoint keys.
    returns a single dictionary that has the union of these keys.
    """
    return dict(chain(*[d.items() for d in args]))


class DatasetBase(Dataset):
    """
    A dataset basically supports iteration over all the examples
    it contains. We currently have 3 datasets inheriting this base
    for 3 types of corpus respectively: "text", "img", "audio".

    Internally it initializes an `torchtext.data.Dataset` object with
    the following attributes:

     `examples`: a sequence of `torchtext.data.Example` objects.
     `fields`: a dictionary associating str keys with `torchtext.data.Field`
        objects, and not necessarily having the same keys as the input fields.
    """
    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 dynamic_dict=False, filter_pred=None):
        # self.src_vocabs: mutated in dynamic_dict, used in translation.py
        self.src_vocabs = []

        # another question: why is the Dataset constructor used in preprocess
        # and translate but not train?

        if tgt_examples_iter is not None:
            examples_iter = (_join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        if dynamic_dict:
            examples_iter = (self._dynamic_dict(ex) for ex in examples_iter)

        fields = list(fields.items())
        # there's a problem here if the dataset doesn't have a tgt at
        # translation time
        example_values = ([ex[k] for k, v in fields] for ex in examples_iter)

        examples = [Example.fromlist(ev, fields) for ev in example_values]

        super(DatasetBase, self).__init__(examples, fields, filter_pred)

    @classmethod
    def make_examples(cls, path, **kwargs):
        """
        path: location of a corpus file
        remaining arguments are passed to _make_iterator_from_file and
        _make_examples
        returns an iterator of dictionaries, one for each example in the corpus
        file
        """
        for i, line in enumerate(cls._make_iterator_from_file(path, **kwargs)):
            example = cls._make_example(line, **kwargs)
            example['indices'] = i
            yield example

    @classmethod
    def _make_iterator_from_file(cls, path, **kwargs):
        return NotImplemented

    @classmethod
    def _make_example(cls, *args, **kwargs):
        return NotImplemented

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, _d):
        self.__dict__.update(_d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(DatasetBase, self).__reduce_ex__()

    def _dynamic_dict(self, example):
        """
        returns an updated version of the example with "src_map" and
        "alignment" values.
        This method also mutates the instance's self.src_vocabs attribute.
        """
        # if using dynamic dict, extra fields are created.
        # All the processing happens in this method, but it shouldn't.
        src = example["src"]
        src_vocab = Vocab(Counter(src), specials=[UNK_WORD, PAD_WORD])
        self.src_vocabs.append(src_vocab)
        # Mapping source tokens to indices in the dynamic dict.
        src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
        example["src_map"] = src_map

        if "tgt" in example:
            mask = torch.LongTensor(
                [0] + [src_vocab.stoi[w] for w in example["tgt"]] + [0])
            example["alignment"] = mask
        return example
