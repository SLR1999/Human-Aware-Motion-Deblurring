import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, ConvTranspose2d, Sigmoid
from resnet_unit import ResidualBlock

class Encoder(Module):   
    def __init__(self):
        super(Encoder, self).__init__()

        self.layers = Sequential(
            Conv2d(6, 32, kernel_size=5, stride=1, padding=2),
            ResidualBlock(32, 32, apply_activation=True),
            ResidualBlock(32, 32, apply_activation=True),
            ResidualBlock(32, 32, apply_activation=True),

            Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            ResidualBlock(64, 64, apply_activation=True),
            ResidualBlock(64, 64, apply_activation=True),
            ResidualBlock(64, 64, apply_activation=True),

            Conv2d(64, 32, kernel_size=5, stride=2, padding=2),
            ResidualBlock(128, 128, apply_activation=True),
            ResidualBlock(128, 128, apply_activation=True),
            ResidualBlock(128, 128, apply_activation=True),
        )


    # Defining the forward pass    
    def forward(self, x):
        x = self.layers(x)
        return x