import time
import math
import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from net import *
from utils import *


mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
tsf = {
    'train': transforms.Compose(
    [
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees = 10, translate = [0.1, 0.1], scale = [0.9, 1.1]),
    transforms.ToTensor(),
    Normalize
    ]),
    'test': transforms.Compose(
    [
    transforms.ToTensor(),
    Normalize
    ])
}

trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = True,
                                download=True, transform = tsf['train'])
testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = False,
                                download=True, transform = tsf['test'])

x_train = trainset.tr
print(x_train)