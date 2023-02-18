import torch
import torch.nn as nn
import torch.nn.functional as F


class vgg19 (nn.Module):
    def __init__(self):
        super(vgg19, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block1 = self._make_block(3, 64, 1)
        self.block2 = self._make_block(64, 128, 1)
        self.block3 = self._make_block(128, 256, 3)
        self.block4 = self._make_block(256, 512, 3)
        self.block5 = self._make_block(512, 512, 3)

    def _make_block(self, in_channels, out_channels, block_order):
        vgg_in_conv = [nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True, stride=1), nn.ReLU()]
        vgg_conv = [nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels, kernel_size=3, padding=1, bias=True, stride=1), nn.ReLU()]
        #vgg_maxpooling = [nn.MaxPool2d(kernel_size=2, stride=2)]
        vgg_block = vgg_in_conv + (vgg_conv * block_order) 
        return nn.Sequential(*vgg_block)

    def forward(self, x):
        x1 = self.block1(x)
        x1_copy = x1
        x1 = self.maxpooling(x1)
        x2 = self.block2(x1)
        x2_copy = x2
        x2 = self.maxpooling(x2)
        x3 = self.block3(x2)
        x3_copy = x3
        x3 = self.maxpooling(x3)
        x4 = self.block4(x3)
        x4_copy = x4
        x4 = self.maxpooling(x4)
        x5 = self.block5(x4)
        #x5 = self.maxpooling(x5)
        return x1_copy, x2_copy, x3_copy, x4_copy, x5


def test_vgg():
    weight = torch.load(r'C:\Double U net\vgg19-dcbb9e9d.pth')
    print(weight.keys())
    model = vgg19()
    new_dict = model.state_dict()
    for i in range(32):
        new_dict[list(new_dict.keys())[i]] = weight[list(weight.keys())[i]]
    model.load_state_dict(new_dict)
    # print(model.state_dict().keys())
    for i in range(32):
        print(weight[list(weight.keys())[i]] ==
              new_dict[list(new_dict.keys())[i]])


if __name__ == '__main__':
    test_vgg()
