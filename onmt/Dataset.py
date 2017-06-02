from __future__ import division

import math
import torch
from torch.autograd import Variable

import onmt


class Dataset(object):
    def __init__(self, srcData, tgtData, batchSize, cuda,
                 volatile=False, data_type="text"):
        self.src = srcData
        self._type = data_type
        if tgtData:
            self.tgt = tgtData
            assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src)/batchSize)
        self.volatile = volatile

    def _batchify(self, data, align_right=False,
                  include_lengths=False, dtype="text"):
        if dtype == "text":
            lengths = [x.size(0) for x in data]
            max_length = max(lengths)
            out = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
            for i in range(len(data)):
                data_length = data[i].size(0)
                offset = max_length - data_length if align_right else 0
                out[i].narrow(0, offset, data_length).copy_(data[i])
            if include_lengths:
                return out, lengths
            else:
                return out
        elif dtype == "img":
            heights = [x.size(1) for x in data]
            max_height = max(heights)
            widths = [x.size(2) for x in data]
            max_width = max(widths)

            out = data[0].new(len(data), 3, max_height, max_width).fill_(0)
            for i in range(len(data)):
                data_height = data[i].size(1)
                data_width = data[i].size(2)
                height_offset = max_height - data_height if align_right else 0
                width_offset = max_width - data_width if align_right else 0
                out[i].narrow(1, height_offset, data_height) \
                      .narrow(2, width_offset, data_width).copy_(data[i])
            return out, widths

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, lengths = self._batchify(
            self.src[index*self.batchSize:(index+1)*self.batchSize],
            align_right=False, include_lengths=True, dtype=self._type)

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index*self.batchSize:(index+1)*self.batchSize],
                dtype="text")
        else:
            tgtBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        batch = (zip(indices, srcBatch) if tgtBatch is None
                 else zip(indices, srcBatch, tgtBatch))
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            indices, srcBatch = zip(*batch)
        else:
            indices, srcBatch, tgtBatch = zip(*batch)

        def wrap(b, dtype="text"):
            if b is None:
                return b
            b = torch.stack(b, 0)
            if dtype == "text":
                b = b.t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)
        lengths = Variable(lengths, volatile=self.volatile)
        return (wrap(srcBatch, self._type), lengths), \
            wrap(tgtBatch, "text"), indices

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])
