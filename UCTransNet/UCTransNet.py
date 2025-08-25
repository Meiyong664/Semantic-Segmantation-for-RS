# the model file is the UCTransNet refer to the research artical 
#   UCTransNet: Rethinking the Skip Connections in U-Net from a Channel-Wise Perspective with Transformer
#   https://ojs.aaai.org/index.php/AAAI/article/view/20144

import torch
import torch.nn as nn
import timm 

class Downsample(nn.Module):
    def __init__(self, conv_nums, in_ch, out_ch = None):
        super().__init__()

        if out_ch is None:
            out_ch = in_ch
        else:
            self.norm = nn.BatchNorm2d(out_ch)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.convs = []
        for _ in range(conv_nums):
            self.convs.append(ConvBNReLU(in_ch, out_ch))

    def forward(self, x):

        x = self.maxpool(x)
        x = self.convs(x)

        return x
    
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3, stride = 1, pad = 1):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad)
        self.BN = nn.BatchNorm2d(out_ch)
        self.Relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.Relu(self.BN(self.conv(x)))

        return x
    
class CCA(nn.Module):
    """
    input:  feature maps from the CCT module, B x (H*W / P^2) x C_i
            the Decoder feature from Decoder_i, B x C x H x W

    O_i upsample+conv -> GAP -> linear+sigmod 
    D_i GAP -> linear+sigmod                   add two -> multi O_i
    """
    def __init__(self):
        super().__init__()

        self.mlp_g = nn.Sequential(

        )
