from encoder import Encoder
from attention_module import Attention
from bgdecoder import BGDecoder
from fgdecoder import FGDecoder
import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, ConvTranspose2d, Sigmoid
from resnet_unit import ResidualBlock

class PDecoder(Module):   
    def __init__(self):
        super(PDecoder, self).__init__()

        self.layer1 = Sequential(
            Conv2d(128*3, 128, kernel_size = 1, stride = 1),
            ResidualBlock(128, 128, apply_activation=True),
            ResidualBlock(128, 128, apply_activation=True),
            ResidualBlock(128, 128, apply_activation=True),
            ConvTranspose2d(128, 64, kernel_size = 5, stride = 2, padding = 2, output_padding = 1, dilation = 1),
        )

        self.layer2 = Sequential(
            Conv2d(64*3, 64, kernel_size = 1, stride = 1),
            ResidualBlock(64, 64, apply_activation=True),
            ResidualBlock(64, 64, apply_activation=True),
            ResidualBlock(64, 64, apply_activation=True),
            ConvTranspose2d(64, 48, kernel_size = 5, stride = 2, padding = 2, output_padding = 1, dilation = 1),
        )


        self.layer3 = Sequential(
            Conv2d(32*2 + 48, 48, kernel_size = 1, stride = 1),
            ResidualBlock(48, 48, apply_activation=True),
            ResidualBlock(48, 48, apply_activation=True),
            ResidualBlock(48, 48, apply_activation=True),
            ConvTranspose2d(48, 1, kernel_size = 5, stride = 1, padding = 3, output_padding = 2, dilation = 1),
        )

        


    # Defining the forward pass    
    def forward(self, encoder_output, fg_output, bg_output, x):
        x = torch.cat([encoder_output, fg_output.layer1, bg_output.layer1], dim = 0)
        x = self.layer1(x)
        x = torch.cat([x, fg_output.layer2, bg_output.layer2], dim = 0)
        x = self.layer2(x)
        x = torch.cat([x, fg_output.layer3, bg_output.layer3], dim = 0)
        x = self.layer3(x)
        return x