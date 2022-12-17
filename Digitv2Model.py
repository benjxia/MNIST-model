import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np


class Digitv2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=5,
            kernel_size=3,
            padding="same",
            padding_mode="zeros",
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )
        self.conv2 = nn.Conv2d(
            in_channels=5,
            out_channels=10,
            kernel_size=3,
            padding="same",
            padding_mode="zeros",
        )

        # Don't ask why there are 2 pools, I don't know why I did that either
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.l1 = nn.Linear(int(10 * 28 * 28 // 16), 3 * 10)
        self.l2 = nn.Linear(3 * 10, 10)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(-1, int(10 * 28 * 28 // 16))
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
