import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, ConvTranspose2d, Sigmoid

class Attention(Module):   
    def __init__(self):
        super(Attention, self).__init__()

        self.encoder_layers = Sequential(
            Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder_layers = Sequential(
            ConvTranspose2d(64,32,kernel_size = 3,stride = 2,padding = 2,output_padding = 1,dilation=2),
            ReLU(inplace=True),

            ConvTranspose2d(32,16,kernel_size = 3,stride = 2,padding = 2,output_padding = 1,dilation=2),
            ReLU(inplace=True),

            ConvTranspose2d(16,3,kernel_size = 3,stride = 2,padding = 2,output_padding = 1,dilation=2),
            ReLU(inplace=True),
        )

        self.prediction_map = Sequential(
            Conv2d(3, 1, kernel_size=1),
            Sigmoid(),
        )


    # Defining the forward pass    
    def forward(self, x):
        x = self.encoder_layers(x)
        x = self.decoder_layers(x)
        x = self.prediction_map(x)
        return x

# test
# img = torch.rand((1,3,16,16))
# attention_module = Attention()
# attention_map_fg = attention_module(img)
# print(attention_map_fg)