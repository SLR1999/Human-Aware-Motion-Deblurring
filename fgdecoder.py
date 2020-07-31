import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, ConvTranspose2d, Sigmoid
from resnet_unit import ResidualBlock

class FGDecoder(Module):   
    def __init__(self):
        super(FGDecoder, self).__init__()
        self.layer1 = Sequential(
            ResidualBlock(128, 128, apply_activation=True),
            ResidualBlock(128, 128, apply_activation=True),
            ResidualBlock(128, 128, apply_activation=True),
            ConvTranspose2d(128, 64, kernel_size = 5, stride = 2, padding = 2, output_padding = 1, dilation = 1),
        )

        self.layer2 = Sequential(
            ResidualBlock(64, 64, apply_activation=True),
            ResidualBlock(64, 64, apply_activation=True),
            ResidualBlock(64, 64, apply_activation=True),
            ConvTranspose2d(64, 32, kernel_size = 5, stride = 2, padding = 2, output_padding = 1, dilation = 1),
        )

        self.layer3 = Sequential(
            ResidualBlock(32, 32, apply_activation=True),
            ResidualBlock(32, 32, apply_activation=True),
            ResidualBlock(32, 32, apply_activation=True),
            ConvTranspose2d(32, 1, kernel_size = 5, stride = 1, padding = 3, output_padding = 2, dilation = 1),
        )


    # Defining the forward pass    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x