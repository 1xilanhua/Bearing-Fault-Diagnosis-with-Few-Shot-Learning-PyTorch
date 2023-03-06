import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, depth_multiplier):
        super(SeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels * depth_multiplier, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels * depth_multiplier, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.convnet = nn.Sequential(
            SeparableConv1D(2, 14, kernel_size=64, stride=8, depth_multiplier=29),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            SeparableConv1D(14, 32, kernel_size=3, stride=1, depth_multiplier=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            SeparableConv1D(32, 64, kernel_size=2, stride=1, depth_multiplier=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            SeparableConv1D(64, 64, kernel_size=3, stride=1, depth_multiplier=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SeparableConv1D(64, 64, kernel_size=3, stride=1, depth_multiplier=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128, 100),
            nn.Sigmoid(),
            nn.Linear(100, 100),
            nn.Sigmoid()
        )

        self.l1_layer = LambdaLayer(lambda tensors: torch.abs(tensors[0] - tensors[1]))
        self.dropout = nn.Dropout(p=0.5)
        self.prediction = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        encoded_l = self.convnet(x1)
        encoded_r = self.convnet(x2)

        L1_distance = self.l1_layer([encoded_l, encoded_r])
        D1_layer = self.dropout(L1_distance)
        prediction = self.prediction(D1_layer)
        return prediction

class WDCNN(nn.Module):
    def __init__(self, input_shape):
        super(WDCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=16, kernel_size=64, stride=9, padding=32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (input_shape[0] // 256), 100)
        self.sig1 = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)  # transpose to NCHW format
        out = self.conv1(x)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = self.pool3(out)
        out = self.conv4(out)
        out = F.relu(out)
        out = self.pool4(out)
        out = self.conv5(out)
        out = F.relu(out)
        out = self.pool5(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.sig1(out)
        return out