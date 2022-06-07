import torch
import torch.nn as nn


# Define  deep UResNet model
class DeepUResnet(nn.Module):
    def __init__(self, nrfilters=32):
        super(DeepUResnet, self).__init__()
        self.convin = torch.nn.Conv2d(1, nrfilters, kernel_size=[3, 3], padding='same', padding_mode='replicate',
                                      bias='True')
        self.conv1d = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv2d = torch.nn.Conv2d(nrfilters, 2 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv3d = torch.nn.Conv2d(2 * nrfilters, 2 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv4d = torch.nn.Conv2d(2 * nrfilters, 4 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv5d = torch.nn.Conv2d(4 * nrfilters, 4 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv6d = torch.nn.Conv2d(4 * nrfilters, 8 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv7d = torch.nn.Conv2d(8 * nrfilters, 8 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv8d = torch.nn.Conv2d(8 * nrfilters, 16 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv1u = torch.nn.Conv2d(16 * nrfilters, 8 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv2u = torch.nn.Conv2d(16 * nrfilters, 8 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv3u = torch.nn.Conv2d(8 * nrfilters, 8 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv4u = torch.nn.Conv2d(8 * nrfilters, 4 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv5u = torch.nn.Conv2d(8 * nrfilters, 4 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv6u = torch.nn.Conv2d(4 * nrfilters, 4 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv7u = torch.nn.Conv2d(4 * nrfilters, 2 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv8u = torch.nn.Conv2d(4 * nrfilters, 2 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv9u = torch.nn.Conv2d(2 * nrfilters, 2 * nrfilters, kernel_size=[3, 3], padding='same',
                                      padding_mode='replicate', bias='True')
        self.conv10u = torch.nn.Conv2d(2 * nrfilters, nrfilters, kernel_size=[3, 3], padding='same',
                                       padding_mode='replicate', bias='True')
        self.conv11u = torch.nn.Conv2d(2 * nrfilters, nrfilters, kernel_size=[3, 3], padding='same',
                                       padding_mode='replicate', bias='True')
        self.conv12u = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[3, 3], padding='same',
                                       padding_mode='replicate', bias='True')
        self.convdeep1 = torch.nn.Conv2d(8 * nrfilters, 1, kernel_size=[3, 3], padding='same', padding_mode='replicate',
                                         bias='True')
        self.convdeep2 = torch.nn.Conv2d(4 * nrfilters, 1, kernel_size=[3, 3], padding='same', padding_mode='replicate',
                                         bias='True')
        self.convdeep3 = torch.nn.Conv2d(2 * nrfilters, 1, kernel_size=[3, 3], padding='same', padding_mode='replicate',
                                         bias='True')
        self.convout = torch.nn.Conv2d(nrfilters, 1, kernel_size=[3, 3], padding='same', padding_mode='replicate',
                                       bias='True')
        self.relu = torch.nn.ReLU()
        self.down = torch.nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    #        self.up = torch.nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

    def forward(self, x):
        y1 = self.relu(self.convin(x))
        y1 = self.relu(self.conv1d(y1))
        #
        y2 = self.down(y1)
        y2 = self.relu(self.conv2d(y2))
        y2 = self.relu(self.conv3d(y2))
        #
        y3 = self.down(y2)
        y3 = self.relu(self.conv4d(y3))
        y3 = self.relu(self.conv5d(y3))
        #
        y4 = self.down(y3)
        y4 = self.relu(self.conv6d(y4))
        y4 = self.relu(self.conv7d(y4))
        #
        y5 = self.down(y4)
        y5 = self.relu(self.conv8d(y5))
        y5 = self.up(y5)
        #
        y6 = self.relu(self.conv1u(y5))
        y6 = torch.cat((y6, y4), 1)
        y6 = self.relu(self.conv2u(y6))
        y6 = self.relu(self.conv3u(y6))
        d1 = self.convdeep1(y6)
        y6 = self.up(y6)
        #
        y7 = self.relu(self.conv4u(y6))
        y7 = torch.cat((y7, y3), 1)
        y7 = self.relu(self.conv5u(y7))
        y7 = self.relu(self.conv6u(y7))
        d2 = self.convdeep2(y7)
        y7 = self.up(y7)
        #
        y8 = self.relu(self.conv7u(y7))
        y8 = torch.cat((y8, y2), 1)
        y8 = self.relu(self.conv8u(y8))
        y8 = self.relu(self.conv9u(y8))
        d3 = self.convdeep3(y8)
        y8 = self.up(y8)
        #
        y9 = self.relu(self.conv10u(y8))
        y9 = torch.cat((y9, y1), 1)
        y9 = self.relu(self.conv11u(y9))
        y9 = self.relu(self.conv12u(y9))
        #
        d1u = self.up(d1)
        d2u = self.up(d2 + d1u)
        d3u = self.up(d3 + d2u)
        #
        y10 = self.convout(y9) + d3u
        #
        out = y10 + x

        return out
