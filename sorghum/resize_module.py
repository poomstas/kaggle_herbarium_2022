# %%
''' 
Implementation of the paper "Learning to Resize Images for Computer Vision Tasks by Talebi et al., (2021)"
Reference on PyTorch Bilinear Interpolation: 
    https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
    https://stackoverflow.com/questions/58676688/how-to-resize-a-pytorch-tensor
'''

# %%
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary
from collections import OrderedDict

# %% Simple example from https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/226557
class LearnToResizeKaggle(nn.Module):
    def __init__(self):
        super(LearnToResizeKaggle, self).__init__()
        self.avgpool = nn.AvgPool2d(2)
        self.downsize = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=7, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(num_features=7),
            nn.ReLU())

    def forward(self, img):
        out = torch.cat((self.avgpool(img), self.downsize(img)), dim=1)
        return out

# %%
class ResBlock(nn.Module):
    ''' To be used in LearnToResize class module '''
    def __init__(self, in_channels=16):
        super(ResBlock, self).__init__()
        self.conv_1         = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1) # k3n16s1
        self.batch_norm_1   = nn.BatchNorm2d(num_features=16)
        self.leaky_relu     = nn.LeakyReLU(negative_slope=0.2)
        self.conv_2         = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1) # k3n16s1
        self.batch_norm_2   = nn.BatchNorm2d(num_features=16)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.batch_norm_1(out)
        out = self.leaky_relu(out)
        out = self.conv_2(out)
        out = self.batch_norm_2(out)
        return out + x

# %%
class LearnToResize(nn.Module):
    ''' Implementation of the model in the paper: "Learning to Resize Images for Computer Vision Tasks by Talebi et al., (2021)"'''
    def __init__(self, num_res_blocks=1, target_size=(224,224)):
        super(LearnToResize, self).__init__()
        self.target_size    = target_size
        self.num_res_blocks = num_res_blocks
        self.conv_1         = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1) # k7n16s1
        self.conv_2         = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1) # k1n16s1
        self.conv_3         = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1) # k3n16s1
        self.conv_4         = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=7, stride=1, padding=3) # k7n3s1
        self.leaky_relu     = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm_1   = nn.BatchNorm2d(num_features=16)
        self.batch_norm_2   = nn.BatchNorm2d(num_features=16)

        if self.num_res_blocks > 0:
            layers = []
            for i in range(self.num_res_blocks):
                layers.append(('ResBlock_'+str(i+1), ResBlock()))
            self.res_layers = nn.Sequential(OrderedDict(layers))
        else:
            self.res_layers = nn.Identity()

    def forward(self, img):
        out = self.conv_1(img)
        out = self.leaky_relu(out)
        out = self.conv_2(out)
        out = self.leaky_relu(out)
        out = self.batch_norm_1(out)
        out_hold = F.interpolate(out, size=self.target_size, mode='bilinear', align_corners=False) # align_corners: https://discuss.pytorch.org/uploads/default/original/2X/6/6a242715685b8192f07c93a57a1d053b8add97bf.png
        out = self.res_layers(out_hold) # If self.num_res_blocks==0, this is an nn.Identity layer
        out = self.conv_3(out)
        out = self.batch_norm_2(out)
        out = self.conv_4(out + out_hold)

        parallel_out = F.interpolate(img, size=self.target_size, mode='bilinear', align_corners=False) # (16, 3, 1024, 1024)

        return out + parallel_out

# %% Test the above class object
if __name__=='__main__':
    # Testing the code used for kaggle (taken from https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/226557)
    module = LearnToResizeKaggle()
    img = torch.rand(16, 1, 2048, 2048, dtype=torch.float32) # batch_size=16
    out = module(img)
    print('Input shape:  ', img.shape)
    print('Output shape: ', out.shape)
    print(summary(module, (1, 2048, 2048), device='cpu'))

    # Testing the paper implementation ("Learning to Resize Images for Computer Vision Tasks")
    img = torch.rand(16, 3, 2048, 2048, dtype=torch.float32) # batch_size=16
    model = LearnToResize(num_res_blocks=1, target_size=(1024, 1024))
    out = model(img)
    summary(model, (3, 2048, 2048), device='cpu')

    # https://discuss.pytorch.org/t/re-using-layers-in-model/48186/2