""" Parts of the U-Net model """

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)
class doubleconv_with_se(nn.Module) :
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(doubleconv_with_se, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SE_Block(c=out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)
class ASPP_module(nn.Module):
    """ASTROUS SPATIAL PYRAMID POOLING"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate, padding):
        super(ASPP_module, self).__init__()
        self.ASPP_Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      dilation=dilation_rate, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.ASPP_Conv(x)


class ASPP(nn.Module):
    """ASPP LAYER"""

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        dilation = [1, 6, 12, 18]
        self.aspp1 = ASPP_module(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=1, dilation_rate=dilation[0], padding=0)
        self.aspp2 = ASPP_module(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=3, dilation_rate=dilation[1], padding=dilation[1])
        self.aspp3 = ASPP_module(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=3, dilation_rate=dilation[2], padding=dilation[2])
        self.aspp4 = ASPP_module(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=3, dilation_rate=dilation[3], padding=dilation[3])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(
                                                 in_channels, 64, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(out_channels*5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[
                           2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels , SE = False):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
            SE_Block(c=out_channels) if SE else None
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True , SE = False):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.out_channels = out_channels
        self.se = SE
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(DoubleConv(in_channels, out_channels),
                SE_Block(c=out_channels) if SE else None)
    def forward(self, x1,x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class Up_de2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True , SE = False):
        super(Up_de2, self).__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.out_channels = out_channels
        self.se = SE
        if bilinear:
            self.up2 = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up2 = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv2 = nn.Sequential(DoubleConv(in_channels,  out_channels),
                SE_Block(c=self.out_channels) if SE else None)
    def forward(self, x1, x2 ,x3):
        x1 = self.up2(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x3, x2, x1], dim=1)
        in_channels_new = x.shape[1]
        if self.se :
            doubleconv = nn.Sequential(DoubleConv(in_channels=in_channels_new, out_channels=self.out_channels),
                SE_Block(c=self.out_channels) if self.se else None)
            return doubleconv(x)
        else :
            return self.conv2(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.out_channels > 1 :
            return self.softmax(self.conv(x))
        else :
            return self.sigmoid(self.conv(x))

class OutConv_v2(nn.Module) :
    def __init__(self, in_channels, out_channels):
        super(OutConv_v2, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.out_channels > 1 :
            x = self.conv(x)
            x = x - torch.max(x)
            return self.softmax(x)
        else :
            return self.sigmoid(self.conv(x))

