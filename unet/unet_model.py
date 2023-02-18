from .unet_layer import *
from .vgg_19 import vgg19
import numpy as np


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128,SE=True)
        self.down2 = Down(128, 256,SE=True)
        self.down3 = Down(256, 512,SE=True)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor,SE=True)
        self.up1 = Up(1024, 512 // factor, bilinear,SE=True)
        self.up2 = Up(512, 256 // factor, bilinear,SE=True)
        self.up3 = Up(256, 128 // factor, bilinear,SE=True)
        self.up4 = Up(128, 64, bilinear,SE=True)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Double_unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Double_unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.vgg_weight = self.get_vgg_weight(
            r'C:\Double U net\vgg19-dcbb9e9d.pth')
        self.inc = DoubleConv(n_channels, 64)
        self.vgg = vgg19()
        new_dict = self.vgg.state_dict()
        for i in range(32):
            new_dict[list(new_dict.keys())[i]] = self.vgg_weight[list(
                self.vgg_weight.keys())[i]]
        self.vgg.load_state_dict(new_dict)
        self.aspp1 = ASPP(512, 64)
        self.aspp2 = ASPP(256, 64)
        self.down1_1 = doubleconv_with_se(3, 32)
        self.down2_1 = Down(32, 64, SE=True)
        self.down3_1 = Down(64, 128, SE=True)
        self.down4_1 = Down(128, 256, SE=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.MaxPool2d(2)
        self.up_conv_1 = doubleconv_with_se(64+512, 256)
        self.up_conv_2 = doubleconv_with_se(256+256, 128)
        self.up_conv_3 = doubleconv_with_se(128+128, 64)
        self.up_conv_4 = doubleconv_with_se(64+64, 32)
        self.up_conv_5 = doubleconv_with_se(64+512+256, 256)
        self.up_conv_6 = doubleconv_with_se(256+256+128, 128)
        self.up_conv_7 = doubleconv_with_se(128+128+64, 64)
        self.up_conv_8 = doubleconv_with_se(64+64+32, 32)
        self.outc1 = OutConv(32, 1)
        self.outc2 = OutConv(32, 1)

    def forward(self, x):
        en1, en2, en3, en4, en5 = self.vgg(x)
        de1_input = self.aspp1(en5)
        de1_input = self.upsample(de1_input)
        de1_input = torch.cat((de1_input, en4), dim=1)
        de1_output = self.up_conv_1(de1_input)
        de2_input = self.upsample(de1_output)
        de2_input = torch.cat((de2_input, en3), dim=1)
        de2_output = self.up_conv_2(de2_input)
        de3_input = self.upsample(de2_output)
        de3_input = torch.cat((de3_input, en2), dim=1)
        de3_output = self.up_conv_3(de3_input)
        de4_input = self.upsample(de3_output)
        de4_input = torch.cat((de4_input, en1), dim=1)
        de4_output = self.up_conv_4(de4_input)
        output1 = self.outc1(de4_output)
        x = torch.mul(x, output1)
        en1_2 = self.down1_1(x)
        en2_2 = self.down2_1(en1_2)
        en3_2 = self.down3_1(en2_2)
        en4_2 = self.down4_1(en3_2)
        en5_2 = self.downsample(en4_2)
        de5_input = self.aspp2(en5_2)
        #de5_input = self.upsample(de5_input)
        # print(de5_input.shape)
        # print(en4.shape)
        # print(en4_2.shape)
        # en4_2 = self.upsample(en4_2)
        # en3_2 = self.upsample(en3_2)
        # en2_2 = self.upsample(en2_2)
        # en1_2 = self.upsample(en1_2)
        de5_input = self.upsample(de5_input)
        de5_input = torch.cat((de5_input, en4, en4_2), dim=1)
        de5_output = self.up_conv_5(de5_input)
        de6_input = self.upsample(de5_output)
        de6_input = torch.cat((de6_input, en3, en3_2), dim=1)
        de6_output = self.up_conv_6(de6_input)
        de7_input = self.upsample(de6_output)
        de7_input = torch.cat((de7_input, en2, en2_2), dim=1)
        de7_output = self.up_conv_7(de7_input)
        de8_input = self.upsample(de7_output)
        de8_input = torch.cat((de8_input, en1, en1_2), dim=1)
        de8_output = self.up_conv_8(de8_input)

        output2 = self.outc2(de8_output)
        output = torch.cat((output1, output2), 1)
        return output

    def get_vgg_weight(self, weight_path):
        pretrained_dict = torch.load(weight_path)
        return pretrained_dict


class Double_unet_multipleclass(nn.Module):
    def __init__(self, n_channels, n_classes, label_values, DEVICE, bilinear=False):
        super(Double_unet_multipleclass, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.color_values = label_values
        self.DEVICE = DEVICE
        self.bilinear = bilinear
        self.vgg = vgg19()
        self.vgg_weight = self.get_vgg_weight(
            r'C:\Double U net\vgg19-dcbb9e9d.pth')
        new_dict = self.vgg.state_dict()
        for i in range(32):
            new_dict[list(new_dict.keys())[i]] = self.vgg_weight[list(
                self.vgg_weight.keys())[i]]
        self.vgg.load_state_dict(new_dict)
        self.inc = DoubleConv(n_channels, 64)
        self.aspp1 = ASPP(512, 64)
        self.aspp2 = ASPP(256, 64)
        self.down1_1 = doubleconv_with_se(3, 32)
        self.down2_1 = Down(32, 64, SE=True)
        self.down3_1 = Down(64, 128, SE=True)
        self.down4_1 = Down(128, 256, SE=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.MaxPool2d(2)
        self.up_conv_1 = doubleconv_with_se(64+512, 256)
        self.up_conv_2 = doubleconv_with_se(256+256, 128)
        self.up_conv_3 = doubleconv_with_se(128+128, 64)
        self.up_conv_4 = doubleconv_with_se(64+64, 32)
        self.up_conv_5 = doubleconv_with_se(64+512+256, 256)
        self.up_conv_6 = doubleconv_with_se(256+256+128, 128)
        self.up_conv_7 = doubleconv_with_se(128+128+64, 64)
        self.up_conv_8 = doubleconv_with_se(64+64+32, 32)
        self.outc1 = OutConv_v2(32, n_classes)
        self.outc2 = OutConv_v2(32, n_classes)

    def forward(self, x):
        en1, en2, en3, en4, en5 = self.vgg(x)
        de1_input = self.aspp1(en5)
        de1_input = self.upsample(de1_input)
        de1_input = torch.cat((de1_input, en4), dim=1)
        de1_output = self.up_conv_1(de1_input)
        de2_input = self.upsample(de1_output)
        de2_input = torch.cat((de2_input, en3), dim=1)
        de2_output = self.up_conv_2(de2_input)
        de3_input = self.upsample(de2_output)
        de3_input = torch.cat((de3_input, en2), dim=1)
        de3_output = self.up_conv_3(de3_input)
        de4_input = self.upsample(de3_output)
        de4_input = torch.cat((de4_input, en1), dim=1)
        de4_output = self.up_conv_4(de4_input)
        output1 = self.outc1(de4_output)
        # print(output1)
        x = output1.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        colour_codes = np.array(self.color_values)
        x =np.transpose(
            colour_codes[x.astype(int)], (0, 3, 1, 2))
        x = x/255.0
        x = torch.tensor(x).float().to(self.DEVICE)
        en1_2 = self.down1_1(x)
        en2_2 = self.down2_1(en1_2)
        en3_2 = self.down3_1(en2_2)
        en4_2 = self.down4_1(en3_2)
        en5_2 = self.downsample(en4_2)
        de5_input = self.aspp2(en5_2)
        #de5_input = self.upsample(de5_input)
        # print(de5_input.shape)
        # print(en4.shape)
        # print(en4_2.shape)
        # en4_2 = self.upsample(en4_2)
        # en3_2 = self.upsample(en3_2)
        # en2_2 = self.upsample(en2_2)
        # en1_2 = self.upsample(en1_2)
        de5_input = self.upsample(de5_input)
        de5_input = torch.cat((de5_input, en4, en4_2), dim=1)
        de5_output = self.up_conv_5(de5_input)
        de6_input = self.upsample(de5_output)
        de6_input = torch.cat((de6_input, en3, en3_2), dim=1)
        de6_output = self.up_conv_6(de6_input)
        de7_input = self.upsample(de6_output)
        de7_input = torch.cat((de7_input, en2, en2_2), dim=1)
        de7_output = self.up_conv_7(de7_input)
        de8_input = self.upsample(de7_output)
        de8_input = torch.cat((de8_input, en1, en1_2), dim=1)
        de8_output = self.up_conv_8(de8_input)
        output2 = self.outc2(de8_output)
        # print(output2)
        return output1, output2

    def get_vgg_weight(self, weight_path):
        pretrained_dict = torch.load(weight_path)
        return pretrained_dict


class unet_withvgg(nn.Module):
    def __init__(self, n_channels, n_classes, label_values, DEVICE, bilinear=False):
        super(unet_withvgg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.color_values = label_values
        self.DEVICE = DEVICE
        self.bilinear = bilinear
        self.vgg = vgg19()
        self.vgg_weight = self.get_vgg_weight(
            r'C:\Double U net\vgg19-dcbb9e9d.pth')
        new_dict = self.vgg.state_dict()
        for i in range(32):
            new_dict[list(new_dict.keys())[i]] = self.vgg_weight[list(
                self.vgg_weight.keys())[i]]
        self.vgg.load_state_dict(new_dict)
        self.inc = DoubleConv(n_channels, 64)
        self.aspp1 = ASPP(512, 64)
        self.aspp2 = ASPP(256, 64)
        self.down1_1 = doubleconv_with_se(3, 32)
        self.down2_1 = Down(32, 64, SE=True)
        self.down3_1 = Down(64, 128, SE=True)
        self.down4_1 = Down(128, 256, SE=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_conv_1 = doubleconv_with_se(64+512, 256)
        self.up_conv_2 = doubleconv_with_se(256+256, 128)
        self.up_conv_3 = doubleconv_with_se(128+128, 64)
        self.up_conv_4 = doubleconv_with_se(64+64, 32)
        self.up_conv_5 = doubleconv_with_se(64+512+256, 256)
        self.up_conv_6 = doubleconv_with_se(256+256+128, 128)
        self.up_conv_7 = doubleconv_with_se(128+128+64, 64)
        self.up_conv_8 = doubleconv_with_se(64+64+32, 32)
        self.outc1 = OutConv_v2(32, n_classes)
        self.outc2 = OutConv_v2(32, n_classes)

    def forward(self, x):
        en1, en2, en3, en4, en5 = self.vgg(x)
        de1_input = self.aspp1(en5)
        de1_input = self.upsample(de1_input)
        de1_input = torch.cat((de1_input, en4), dim=1)
        de1_output = self.up_conv_1(de1_input)
        de2_input = self.upsample(de1_output)
        de2_input = torch.cat((de2_input, en3), dim=1)
        de2_output = self.up_conv_2(de2_input)
        de3_input = self.upsample(de2_output)
        de3_input = torch.cat((de3_input, en2), dim=1)
        de3_output = self.up_conv_3(de3_input)
        de4_input = self.upsample(de3_output)
        de4_input = torch.cat((de4_input, en1), dim=1)
        de4_output = self.up_conv_4(de4_input)
        output1 = self.outc1(de4_output)

        return output1

    def get_vgg_weight(self, weight_path):
        pretrained_dict = torch.load(weight_path)
        return pretrained_dict
