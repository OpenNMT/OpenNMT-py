"""
This file defines the model architecture and the functionalities of a language model.
It re-uses the components of the decoder.
"""
from __future__ import division

import math
import torch
import torch.nn as nn
import onmt
from onmt.Models import StackedLSTM, StackedGRU
from torch.autograd import Variable
from train_lm import memoryEfficientLoss, MLECriterion

class LM(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        input_size = opt.word_vec_size

        super(LM, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)

        stackedCell = StackedLSTM if opt.rnn_type == "LSTM" else StackedGRU
        self.rnn_type = opt.rnn_type
        self.rnn = stackedCell(opt.layers, input_size,
                               opt.rnn_size, opt.dropout)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs is not None:
            pretrained = torch.load(opt.pre_word_vecs)
            self.word_lut.weight.data.copy_(pretrained)

    def init_rnn_state(self, batch_size, cuda):
        def get_variable():
            v = torch.zeros(self.layers, batch_size, self.hidden_size)
            if cuda:
                v = Variable(v.cuda(), requires_grad=False)
            else:
                v = Variable(v, requires_grad=False)
            return v

        if self.rnn_type == 'LSTM':
            state = (get_variable(), get_variable())
        elif self.rnn_type == 'GRU':
            state = get_variable()
        else:
            raise NotImplementedError("Not valid rnn_type: %s" % self.rnn_type)

        return state

    def forward(self, input):
        n_steps, batch_size = input.size()
        emb = self.word_lut(input)
        hidden = self.init_rnn_state(batch_size, emb.is_cuda)
        outputs = []
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            output, hidden = self.rnn(emb_t, hidden)
            output = self.dropout(output)

            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs

class LMPredictor(object):
    def __init__(self, opt):
        self.opt = opt
        checkpoint = torch.load(opt.model,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        self.src_dict = checkpoint['dicts']['src']
        model = LM(opt, self.src_dict)
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, self.src_dict.size()),
            nn.LogSoftmax())
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
        self.criterion = MLECriterion(self.src_dict.size(), opt.cuda)

        if opt.cuda:
            model.cuda()
            generator.cuda()
        else:
            model.cpu()
            generator.cpu()

        model.generator = generator

        self.model = model
        self.model.eval()


    def _getBatchSize(self, batch):
        if self._type == "monotext":
            return batch.size(1)
        else:
            return None

    def buildData(self, data):
        # This needs to be the same as preprocess.py.
        srcData = [self.src_dict.convertToIdx(sent,
                                              onmt.Constants.UNK_WORD,
                                              onmt.Constants.BOS_WORD,
                                              onmt.Constants.EOS_WORD
                                              )
                   for sent in data]

        return onmt.Dataset(srcData, None, self.opt.batch_size,
                            self.opt.cuda, volatile=True,
                            data_type='monotext')

    def buildTargetTokens(self, pred):
        tokens = self.src_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS

        return tokens

    def eval(self, output, data):
        loss, _, _ = memoryEfficientLoss(output, data,
                                         self.model.generator,
                                         self.criterion,
                                         self.opt.max_generator_batches,
                                         True)
        return loss

    def computePerplexity(self, data):
        """
        Compute the perplexity on the dataset data
        :param data: list of lists of words
        :return: the perplexity
        """
        total_loss, num_words = 0, 0
        dataset = self.buildData(data)
        for i in range(len(dataset)):
            batch = dataset[i][:-1][0]

            srcBatch = batch[0][:-1]
            outputs = self.model(srcBatch)
            # Exclude <s> from targets.
            targets = batch[0][1:]

            loss = self.eval(outputs, targets)
            total_loss += loss
            num_words += targets.data.ne(onmt.Constants.PAD).sum()

        loss = total_loss / num_words
        return math.exp(min(loss, 100))

    def generate(self, starter):
        raise NotImplementedError("Still to implement!")