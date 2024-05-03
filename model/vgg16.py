import torch
import torch.nn as nn


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        h = self.relu1(self.conv1_1(X))
        h = self.relu2(self.conv1_2(h))
        relu1_2 = h
        h = self.mp1(h)

        h = self.relu3(self.conv2_1(h))
        h = self.relu4(self.conv2_2(h))
        relu2_2 = h
        h = self.mp2(h)

        return [relu1_2, relu2_2]
