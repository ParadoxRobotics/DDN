# Compute contrastive loss given network output and match/non-match
# Author : Munch Quentin, 2020

import math
from random import randint
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VisualDescriptorNet(torch.nn.Module):
    def __init__(self, descriptorDim, trainingMode):
        super(VisualDescriptorNet, self).__init__()
        # D dimensionnal descriptors
        self.descriptorDim = descriptorDim
        # Get pretrained Resnet18 without last actiavtion layer (softmax)
        self.ResNet = nn.Sequential(*list(models.resnet18(pretrained=trainingMode).children())[:-2])
        # Resnet layer block list
        layerBlock = list(self.ResNet.children())
        # Encoder layer output for the FCN
        self.layerBlock0 = layerBlock[0] # size=(N, 64, x.H/2, x.W/2)
        self.layerBlock1 = layerBlock[1] # size=(N, 64, x.H/2, x.W/2)
        self.layerBlock2 = layerBlock[2] # size=(N, 64, x.H/2, x.W/2)
        self.layerBlock3 = layerBlock[3] # size=(N, 64, x.H/4, x.W/4)
        self.layerBlock4 = layerBlock[4] # size=(N, 64, x.H/4, x.W/4)

        self.layerBlock5 = layerBlock[5] # size=(N, 128, x.H/8, x.W/8)
        self.layerBlock6 = layerBlock[6] # size=(N, 256, x.H/16, x.W/16)
        self.layerBlock7 = layerBlock[7] # size=(N, 512, x.H/32, x.W/32)
        # Decoder layer
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        # Fusion layer
        self.fuseConv = nn.Conv2d(64 + 128 + 256 + 512, self.descriptorDim, 1)
        # activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layerBlock0(x)
        x = self.layerBlock1(x)
        x = self.layerBlock2(x)
        x = self.layerBlock3(x)
        x = self.layerBlock4(x)
        print(x.shape)
        up1 = self.upsample1(x)
        x = self.layerBlock5(x)
        print(x.shape)
        up2 = self.upsample2(x)
        x = self.layerBlock6(x)
        print(x.shape)
        up3 = self.upsample3(x)
        x = self.layerBlock7(x)
        print(x.shape)
        up4 = self.upsample4(x)
        merge = torch.cat([up1, up2, up3, up4], dim=1)
        print(merge.shape)
        out = self.fuseConv(merge)
        out = self.activation(out)
        return out
