import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, apply_activation=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.apply_activation = apply_activation
        
    def forward(self, x):
        """Output size is same as input size"""
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out += residual

        residual = out
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        residual = out
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        if self.apply_activation: out = self.relu(out)
        return out

