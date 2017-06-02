import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable


class ImageEncoder(nn.Module):
    def __init__(self, opt):
        super(ImageEncoder, self).__init__()
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        self.hidden_size = opt.rnn_size // self.num_directions

        self.layer1 = nn.Conv2d(1,   64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer2 = nn.Conv2d(64,  128, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer3 = nn.Conv2d(128, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer4 = nn.Conv2d(256, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer5 = nn.Conv2d(256, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer6 = nn.Conv2d(512, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))

        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(512)

        input_size = 512
        self.rnn = nn.LSTM(input_size, 512,
                           num_layers=opt.layers,
                           dropout=opt.dropout,
                           bidirectional=opt.brnn)
        self.pos_lut = nn.Embedding(100, 512)

    def load_pretrained_vectors(self, opt):
        pass

    def forward(self, input):
        input = input[0]
        batchSize = input.size(0)
        # (batch_size, 64, imgH, imgW)
        # layer 1
        input = F.relu(self.layer1(input[:, 0:1, :, :]-0.5), True)

        # (batch_size, 64, imgH/2, imgW/2)
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))

        # (batch_size, 128, imgH/2, imgW/2)
        # layer 2
        input = F.relu(self.layer2(input), True)

        # (batch_size, 128, imgH/2/2, imgW/2/2)
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))

        #  (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer 3
        # batch norm 1
        input = F.relu(self.batch_norm1(self.layer3(input)), True)

        # (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer4
        input = F.relu(self.layer4(input), True)

        # (batch_size, 256, imgH/2/2/2, imgW/2/2)
        input = F.max_pool2d(input, kernel_size=(1, 2), stride=(1, 2))

        # (batch_size, 512, imgH/2/2/2, imgW/2/2)
        # layer 5
        # batch norm 2
        input = F.relu(self.batch_norm2(self.layer5(input)), True)

        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        input = F.max_pool2d(input, kernel_size=(2, 1), stride=(2, 1))

        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        input = F.relu(self.batch_norm3(self.layer6(input)), True)

        # # (batch_size, 512, H, W)
        # # (batch_size, H, W, 512)
        all_outputs = []
        for row in range(input.size(2)):
            inp = input[:, :, row, :].transpose(0, 2)\
                                     .transpose(1, 2)
            pos_emb = self.pos_lut(
                Variable(torch.cuda.LongTensor(batchSize).fill_(row)))
            with_pos = torch.cat(
                (pos_emb.view(1, pos_emb.size(0), pos_emb.size(1)), inp), 0)
            outputs, hidden_t = self.rnn(with_pos)
            all_outputs.append(outputs)
        out = torch.cat(all_outputs, 0)

        return hidden_t, out
