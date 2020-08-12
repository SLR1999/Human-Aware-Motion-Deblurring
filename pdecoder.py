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
            Conv2d(1, 3, kernel_size=1, stride=1),
        )

        


    # Defining the forward pass    
    def forward(self, encoder_output, fg_branch_input, bg_branch_input, fg_output, bg_output, x):
        fg_l1 = FGDecoder.layer1(fg_branch_input)
        bg_l1 = BGDecoder.layer1(bg_branch_input)
        x = torch.cat([encoder_output, fg_l1, bg_l1], dim = 0)
        x = self.layer1(x)
        fg_l2 = FGDecoder.layer2(fg_l1)
        bg_l2 = BGDecoder.layer2(bg_l1)
        x = torch.cat([x, fg_l2, bg_l2], dim = 0)
        x = self.layer2(x)
        fg_l3 = FGDecoder.layer3(fg_l2)
        bg_l3 = BGDecoder.layer3(bg_l2)
        x = torch.cat([x, fg_l3, bg_l3], dim = 0)
        x = self.layer3(x)
        return x