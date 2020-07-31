from encoder import Encoder
from attention_module import Attention
from bgdecoder import BGDecoder
from fgdecoder import FGDecoder
from pdecoder import PDecoder
import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, ConvTranspose2d, Sigmoid

class HumanAware(Module):   
    def __init__(self):
        super(HumanAware, self).__init__()

        self.upsample = Sequential(
            Conv2d(3, 3, kernel_size=5, stride=4, padding=2),
        )

        self.downsample = Sequential(
            ConvTranspose2d(3,3,kernel_size = 4,stride = 2,padding = 1,output_padding = 0,dilation=1)
        )

        self.attention_module = Attention()

        self.encoder = Encoder()

        self.fgdecoder = FGDecoder()
        self.bgdecoder = BGDecoder()
        self.pdecoder = PDecoder()

    # Defining the forward pass    
    def forward(self, img, prev_img):
        downsampled_img = self.downsample(img)
        upsampled_prev_img = self.upsample(prev_img)
        attention_map_fg = self.attention_module(downsampled_img)
        attention_map_bg = 1 - attention_map_fg
        encoder_input = torch.stack(img,upsampled_prev_img)
        primary_branch_input = self.encoder(encoder_input)
        stacked_fg_attention = attention_map_fg.copy()
        stacked_bg_attention = attention_map_bg.copy()
        for i in range(127) :
            stacked_fg_attention = torch.stack(stacked_fg_attention,attention_map_fg)
            stacked_bg_attention = torch.stack(stacked_bg_attention,attention_map_bg)
        fg_branch_input = torch.mul(primary_branch_input,stacked_fg_attention)
        bg_branch_input = torch.mul(primary_branch_input,stacked_bg_attention)

        fg_decoder_output = self.fgdecoder(fg_branch_input)
        bg_decoder_output = self.bgdecoder(bg_branch_input)

        p_decoder_output = self.pdecoder(primary_branch_input, fg_decoder_output, bg_decoder_output)
        

        return img


