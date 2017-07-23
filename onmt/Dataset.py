from __future__ import division
import math
import torch
from torch.autograd import Variable
import onmt
from onmt.modules import aeq


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
        self.volatile = volatile

        # Upack data.
        self.src = data["src"]
        self.srcFeatures = data.get("src_features")
        self._type = data.get("type", "text")
        if data["tgt"]:
            self.tgt = data["tgt"]
            assert(len(self.src) == len(self.tgt))
            self.tgtFeatures = data.get("tgt_features")
        else:
            self.tgt = None
            self.tgtFeatures = None
        self.alignment = data.get("alignments")
        self.numBatches = math.ceil(len(self.src)/batchSize)

    def __len__(self):
        return self.numBatches

    def _batchifyImages(self, data):
        max_height = max([x.size(1) for x in data])
        widths = [x.size(2) for x in data]
        max_width = max(widths)
        out = data[0].new(len(data), 3, max_height, max_width).fill_(0)
        for i in range(len(data)):
            out[i, :, :data[i].size(1), :data[i].size(2)].copy_(data[i])
        return out, torch.FloatTensor(widths)

    def _batchifyAlignment(self, data, src_len, tgt_len, batch):
        alignment = torch.ByteTensor(tgt_len, batch, src_len).fill_(0)
        for i in range(len(data)):
            alignment[1:data[i].size(1)+1, i, :data[i].size(0)] \
                = data[i].t()
        return alignment.float()

    def _batchify(self, data, features=None):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        num_features = len(features) if features else 0
        out = data[0].new(len(data), max_length, num_features + 1) \
                     .fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            out[i, :data_length, 0].copy_(data[i])
            for j in range(num_features):
                out[i, :data_length, j+1].copy_(features[j][i])
        return out.transpose(0, 1).contiguous(), torch.FloatTensor(lengths)

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        s = index*self.batchSize
        e = s + self.batchSize
        src = self.src[s:e]
        batch_size = len(src)
        features = None
        if self.srcFeatures:
            features = [f[s:e] for f in self.srcFeatures]

        if self._type == "text":
            srcBatch, lengths = self._batchify(src, features=features)
        else:
            srcBatch, lengths = self._batchifyImage(src)

        tgtBatch = None
        if self.tgt:
            tgtBatch, _ = self._batchify(self.tgt[s:e])
            tgtBatch = tgtBatch.squeeze(2)

        # Create an alignment object.
        alignment = None
        if self.alignment:
            alignment = self._batchifyAlignment(self.alignment[s:e],
                                                srcBatch.size(0),
                                                tgtBatch.size(0),
                                                batch_size)

        # Make a `Batch` object.
        batch = Batch(srcBatch, tgtBatch, lengths, None, len(src), alignment)
        batch.reorderByLength()
        batch.wrap(self.volatile, self.cuda)
        return batch

    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])


class Batch(object):
    """
    Object containing a single batch of data points.
    """
    def __init__(self, src, tgt, lengths, indices, batchSize, alignment=None):
        # CHECKS
        s_len, n_batch, n_feats = src.size()
        t_len, n_batch_ = tgt.size()
        aeq(n_batch, n_batch_)
        n_batch_, = lengths.size()
        aeq(n_batch, n_batch_)

        if alignment is not None:
            pass
        # END CHECKS

        self.src = src
        self.tgt = tgt
        self.lengths = lengths
        self.indices = indices
        self.batchSize = batchSize
        self.alignment = alignment

    def wrap(self, volatile, cuda):
        def _wrap(v):
            if v is None:
                return None
            if cuda:
                v = v.cuda()
            v = Variable(v, volatile=volatile)
            return v
        self.lengths = _wrap(self.lengths)
        self.src = _wrap(self.src)
        self.tgt = _wrap(self.tgt)
        self.alignment = _wrap(self.alignment)

    def reorderByLength(self):
        # tgt_len x batch x src_len
        # within batch sorting by decreasing length for variable length rnns
        self.indices = torch.LongTensor(range(self.src.size(1)))

        self.lengths, perm = torch.sort(self.lengths, 0,
                                        descending=True)

        # Reorder all.
        self.indices = self.indices.index_select(0, perm).contiguous()
        self.src = self.src.index_select(1, perm).contiguous()
        if self.tgt is not None:
            self.tgt = self.tgt.index_select(1, perm).contiguous()
        if self.alignment is not None:
            self.alignment = self.alignment.index_select(1, perm).contiguous()

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
                     self.alignment[start:end]
                     if self.alignment is not None else None)
