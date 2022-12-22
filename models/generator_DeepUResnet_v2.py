"""
Created on Wed Nov  9 15:53:55 2022

@author: nly22608
"""

import torch
from torch import nn


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=[3, 3], padding='same', padding_mode='replicate', bias='True')
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=[3, 3], padding='same', padding_mode='replicate', bias='True')
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.25)

    def forward(self, inputs):
        x = self.conv1(inputs)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class build_deepunet(nn.Module):
    def __init__(self):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        """ Bottleneck """
        self.b = conv_block(512, 1024)
        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        """ deep layers """
        self.bd = nn.Conv2d(1024, 1, kernel_size=[3, 3], padding='same', padding_mode='replicate', bias='True')
        self.d1d = nn.Conv2d(512, 1, kernel_size=[3, 3], padding='same', padding_mode='replicate', bias='True')
        self.d2d = nn.Conv2d(256, 1, kernel_size=[3, 3], padding='same', padding_mode='replicate', bias='True')
        self.d3d = nn.Conv2d(128, 1, kernel_size=[3, 3], padding='same', padding_mode='replicate', bias='True')
        """ Unet output """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        """ upsample """
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Deep layers """
        bd = self.bd(b)
        d1d = self.d1d(d1)
        up1 = d1d + self.up(bd)
        d2d = self.d2d(d2)
        up2 = d2d + self.up(up1)
        d3d = self.d3d(d3)
        up3 = d3d + self.up(up2)
        """ Resnet output = input + Unet output + deep layers output """
        outputs = self.outputs(d4) + self.up(up3) + inputs
        return outputs