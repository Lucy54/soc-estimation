"""
Tutorial: https://www.youtube.com/watch?v=bH9Nkg7G8S0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import math
from collections import OrderedDict

torch.set_printoptions(linewidth=150)

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

image, label = train_set[0]
print(image.shape)

plt.imshow(image.squeeze(), cmap='gray')
print(train_set.classes)

in_features = image.numel()
out_features = math.floor(in_features / 2)

out_classes = len(train_set.classes)

# First sequential model
network1 = nn.Sequential(
    nn.Flatten(start_dim=1)
    ,nn.Linear(in_features, out_features) # hidden layer
    ,nn.Linear(out_features, out_classes) # output layer
)

print("NETWORK1[1]", network1[1])

image = image.unsqueeze(0)
print("image.shape: ", image.shape)

layers = OrderedDict([
    ('flat', nn.Flatten(start_dim=1))
    ,('hidden', nn.Linear(in_features, out_features))
    ,('output', nn.Linear(out_features, out_classes))
])

network2 = nn.Sequential(layers)