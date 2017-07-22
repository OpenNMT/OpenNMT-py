from __future__ import division

import math
import torch
from torch.autograd import Variable

import onmt


class Dataset(object):
    """
    Manages dataset creation and usage.

    Example:

        `batch = data[batchnum]`
    """

    def __init__(self, data, batchSize, cuda,
                 volatile=False, data_type="text"):
        """
        Construct a data set

        Args:
            data: dictionary from preprocess.py
            batchSize: Training batchSize to use.
            cuda: Return batches on gpu.
            volatile: use at test time
        """
        self.cuda = cuda
        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src)/batchSize)
        self.volatile = volatile

        # Upack data.
        self.src = data["srcData"]
        self.srcFeatures = data.get("srcFeatures")
        self._type = data.get("type", "text")
        if data["tgtData"]:
            self.tgt = data["tgtData"]
            assert(len(self.src) == len(self.tgt))
            self.tgtFeatures = data.get("tgtFeatures")
        else:
            self.tgt = None
            self.tgtFeatures = None
        self.alignment = data.get("alignment")

    def __len__(self):
        return self.numBatches

    def _batchifyImages(self, data):
        max_height = max([x.size(1) for x in data])
        widths = [x.size(2) for x in data]
        max_width = max(widths)
        out = data[0].new(len(data), 3, max_height, max_width).fill_(0)
        for i in range(len(data)):
            out[i, :, :data[i].size(1), :data[i].size(2)].copy_(data[i])
        return out, widths

    def _batchifyAlignment(self, data, srcBatch, tgtBatch):
        src_len = srcBatch.size(1)
        tgt_len = tgtBatch.size(1)
        batch = tgtBatch.size(0)
        alignment = torch.ByteTensor(tgt_len, batch, src_len).fill_(0)

        for i in range(len(data)):
            alignment[1:data[i].size(1)+1, i, :data[i].size(0)] \
                = data[i].t()
        alignment = alignment.float()
        if self.cuda:
            alignment = alignment.cuda()
        return alignment

    def _batchify(self, data, features=None):
        # Create batches.
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        if features:
            num_features = len(features)
            out = data[0].new(len(data), max_length, num_features + 1)
            assert (len(data) == len(features[0])), \
                ("%s %s" % (data[0].size(), len(features[0])))
        else:
            out = data[0].new(len(data), max_length, 1)

        out.fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            out[i, :data_length, 0].copy_(data[i])
            if features:
                for j in range(num_features):
                    out[i, :data_length, j+1].copy_(features[j][i])

        return out, lengths

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        s = index*self.batchSize
        e = s + self.batchSize
        src = self.src[s:e]

        features = None
        if self.srcFeatures:
            features = [f[s:e] for f in self.srcFeatures]

        if self._type == "text":
            srcBatch, lengths = self._batchify(src, features=features)
            srcBatch = srcBatch.transpose(0, 1).contiguous()
        else:
            srcBatch, lengths = self._batchifyImage(src)

        tgtBatch = None
        if self.tgt:
            tgtBatch, _ = self._batchify(self.tgt[s:e])

        # Create an alignment object.
        alignment = None
        if self.alignment:
            alignment = self._batchifyAlignment(self.alignment[s:e],
                                                srcBatch, tgtBatch)

        # Make a `Batch` object.
        batch = Batch(srcBatch, tgtBatch, lengths, None, len(src), alignment)
        batch.wrap(self.volatile, self.cuda)
        batch.reorderByLength()
        return batch

    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])


class Batch(object):
    """
    Object containing a single batch of data points.
    """
    def __init__(self, src, tgt, lengths, indices, batchSize, alignment=None):
        self.src = src
        self.tgt = tgt
        self.lengths = lengths
        self.indices = indices
        self.batchSize = batchSize
        self.alignment = alignment

    def wrap(self, volatile, cuda):
        def _wrap(v):
            if self.cuda:
                v = v.cuda()
            v = Variable(v, volatile=self.volatile)
            return v
        self.length = _wrap(self.lengths)
        self.src = _wrap(self.src)
        self.tgt = _wrap(self.tgt)

    def reorderByLength(self):
        # tgt_len x batch x src_len
        # within batch sorting by decreasing length for variable length rnns
        self.indices = range(len(self.src.size(1)))
        self.lengths[0], perm = torch.sort(self.lengths[0], 0,
                                           descending=True)

        # Reorder all.
        self.indices = self.indices[:, perm].contiguous()
        self.src = self.src[:, perm].contiguous()
        if self.tgt is not None:
            self.tgt = self.tgt[:, perm].contiguous()
        if self.alignment is not None:
            self.alignment = self.alignment[:, perm].contiguous()

    def words(self):
        return self.src[:, :, 0]

    def features(self, j):
        return self.src[:, :, j+1]

    def truncate(self, start, end):
        """
        Return a batch containing section from start:end.
        """
        return Batch(self.src, self.tgt[start:end],
                     self.lengths, self.indices, self.batchSize,
                     self.alignment[start:end])
