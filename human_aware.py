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

        self.downsample = Sequential(
            Conv2d(3, 3, kernel_size=5, stride=4, padding=2),
        )

        self.upsample = Sequential(
            ConvTranspose2d(3,3,kernel_size = 4,stride = 2,padding = 1,output_padding = 0,dilation=1)
        )

        self.attention_module = Attention()

        self.encoder = Encoder()

        self.fgdecoder = FGDecoder()
        self.bgdecoder = BGDecoder()
        self.pdecoder = PDecoder()

    # Defining the forward pass    
    # def forward(self, img, prev_img):
    def forward(self,img):
        downsampled_img = self.downsample(img)
        # upsampled_prev_img = self.upsample(prev_img)
        attention_map_orig = self.attention_module(img)
        attention_map_fg = self.attention_module(downsampled_img)
        attention_map_bg = 1 - attention_map_fg
        encoder_input = img
        # encoder_input = torch.stack(img,upsampled_prev_img)
        primary_branch_input = self.encoder(encoder_input)
        stacked_fg_attention = attention_map_fg.clone()
        stacked_bg_attention = attention_map_bg.clone()
        for i in range(127) :
            stacked_fg_attention = torch.cat((stacked_fg_attention,attention_map_fg),1)
            stacked_bg_attention = torch.cat((stacked_bg_attention,attention_map_bg),1)
        fg_branch_input = torch.mul(primary_branch_input,stacked_fg_attention)
        bg_branch_input = torch.mul(primary_branch_input,stacked_bg_attention)
        fg_decoder_output, fg_l1, fg_l2, fg_l3 = self.fgdecoder(fg_branch_input)
        bg_decoder_output, bg_l1, bg_l2, bg_l3 = self.bgdecoder(bg_branch_input)
        p_decoder_output = self.pdecoder(primary_branch_input, fg_l1, fg_l2, fg_l3, bg_l1, bg_l2, bg_l3)

        return p_decoder_output, fg_decoder_output, bg_decoder_output, attention_map_orig


