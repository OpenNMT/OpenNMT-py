import torch
import codecs
import onmt


class Dict(object):
    def __init__(self, data=None, lower=False):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}
        self.lower = lower

        # Special entries will not be pruned.
        self.special = []

        if data is not None:
            if type(data) == str:
                self.loadFile(data)
            else:
                self.addSpecials(data)

    def size(self):
        return len(self.idxToLabel)

    def loadFile(self, filename):
        "Load entries from a file."
        for line in codecs.open(filename, 'r', 'utf-8'):
            fields = line.split()
            label = fields[0]
            idx = int(fields[1])
            self.add(label, idx)

    def writeFile(self, filename):
        "Write entries to a file."
        with codecs.open(filename, 'w', 'utf-8') as file:
            for i in range(self.size()):
                label = self.idxToLabel[i]
                file.write('%s %d\n' % (label, i))

        file.close()

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def align(self, other):
        "Find the id of each label in other dict."
        alignment = [onmt.Constants.PAD] * self.size()
        for idx, label in self.idxToLabel.items():
            if label in other.labelToIdx:
                alignment[idx] = other.labelToIdx[label]
        return alignment

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    def addSpecial(self, label, idx=None):
        "Mark this `label` and `idx` as special (i.e. will not be pruned)."
        idx = self.add(label, idx)
        self.special += [idx]

    def addSpecials(self, labels):
        "Mark all labels in `labels` as specials (i.e. will not be pruned)."
        for label in labels:
            self.addSpecial(label)

    def add(self, label, idx=None):
        "Add `label` in the dictionary. Use `idx` as its index if given."
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    def prune(self, size):
        "Return a new dictionary with the `size` most frequent entries."
        if size >= self.size():
            return self

        # Only keep the `size` most frequent entries.
        freq = torch.Tensor(
                [self.frequencies[i] for i in range(len(self.frequencies))])
        _, idx = torch.sort(freq, 0, True)

        newDict = Dict()
        newDict.lower = self.lower

        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])

        for i in idx[:size]:
            newDict.add(self.idxToLabel[i])

        return newDict

    def convertToIdx(self, labels, unkWord, bosWord=None, eosWord=None):
        """
        Convert `labels` to indices. Use `unkWord` if not found.
        Optionally insert `bosWord` at the beginning and `eosWord` at the .
        """
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return torch.LongTensor(vec)

    def convertToLabels(self, idx, stop):
        """
        Convert `idx` to labels.
        If index `stop` is reached, convert it and return.
        """

        labels = []

        for i in idx:
            labels += [self.getLabel(i)]
            if i == stop:
                break

        return labels
