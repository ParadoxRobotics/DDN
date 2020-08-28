# Simple code to evaluate on a single image the network
# Author : Munch Quentin, 2020

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from denseDescriptorNetwork_ import *

#get image and resize it
img = cv2.imread('test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640,480))

plt.imshow(img)
plt.show()

input = torch.reshape(torch.from_numpy(img), (3,640,480))
input = input.view((1, *input.shape)).type(torch.FloatTensor)
print(input.shape)

model = VisualDescriptorNet(512, True)

y, ht = model(input)

for i in range(0,10):
    fHt = ht[0,i,:,:]
    plt.matshow(fHt.detach().numpy())
    plt.show()

for i in range(0,3):
    fy = y[0,i,:,:]
    plt.matshow(fy.detach().numpy())
    plt.show()
