import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, dim):
        super(ImageEncoder, self).__init__()

        self.layer1 = nn.SpatialConvolution(  1,  64, 3, 3, 1, 1, 1, 1)
        self.layer2 = nn.SpatialConvolution( 64, 128, 3, 3, 1, 1, 1, 1)
        self.layer3 = nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)
        self.layer4 = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
        self.layer5 = nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)
        self.layer6 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)

        self.batch_norm1 = nn.SpatialBatchNormalization(256)
        self.batch_norm2 = nn.SpatialBatchNormalization(512)
        self.batch_norm3 = nn.SpatialBatchNormalization(512)
        
    def forward(input):
        
        # input shape: (batch_size, 1, imgH, imgW)
        input = (input - 128.0) / 128.0

        # (batch_size, 64, imgH, imgW)
        # layer 1
        input = F.relu(self.layer1(input), true)

        # (batch_size, 64, imgH/2, imgW/2)
        input = F.max_pool2d(input, 2, 2, 2, 2)

        # (batch_size, 128, imgH/2, imgW/2)
        # layer 2
        input = F.relu(self.layer2(input), true)

        # (batch_size, 128, imgH/2/2, imgW/2/2)
        input = F.max_pool2d(input, 2, 2, 2, 2)

        #  (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer 3
        # batch norm 1
        input = F.relu(self.batch_norm1(self.layer3(input)), true)

        # (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer4
        input = F.relu(self.layer4(input), true)

        # (batch_size, 256, imgH/2/2/2, imgW/2/2)
        input = F.max_pool2d(input, 1, 2, 1, 2, 0, 0)) 

        # (batch_size, 512, imgH/2/2/2, imgW/2/2)
        # layer 5
        # batch norm 2
        input = F.relu(self.batch_norm2(self.layer5(input)), true)

        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        input = F.max_pool2d(input, 2, 1, 2, 1, 0, 0)

        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        input = F.relu(self.batch_norm3(self.layer6(input)), true)

        # # (batch_size, 512, H, W)
        # # (batch_size, H, W, 512)
        # model:add(nn.Transpose({2, 3}, {3,4}))
        #  #H list of (batch_size, W, 512)
        # model:add(nn.SplitTable(1, 3)) 

        return input
