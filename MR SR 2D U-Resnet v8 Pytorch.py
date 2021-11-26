# -*- coding: utf-8 -*-
"""
Created on 8 August 2021
Updated on 21 November 2021 (version 8)

@author: Marcel Breeuwer, Philips & TU/e
"""

import sys
import numpy as np
import os
import time
import pydicom
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchinfo import summary
import warnings

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nUsing {} device".format(device))

# clear GPU cache
torch.cuda.empty_cache()

# suppress warnings
warnings.filterwarnings("ignore")

window_size = 11

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssimloss_initwindow(window_size):
    window = create_window(window_size,1)
    return window

window = ssimloss_initwindow(window_size).to(device) # needed every pass?

def ssimloss(img1, img2):
#    window = ssimloss_initwindow(window_size).to(device) # needed every pass?
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = 1)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = 1)

    mu1_sq  = mu1*mu1
    mu2_sq  = mu2*mu2
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = 1) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = 1) - mu2_sq
    sigma12   = F.conv2d(img1*img2, window, padding = window_size//2, groups = 1) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    # SSIM loss = 1.0 - SSIM(img1,img2)
    ssim_map = 1.0-((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

# function to write to dicom
def write2dicom(inpath, outpath, indata, protname, number):
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    lengths = indata.shape
    lz = lengths[0]
    znum = 0
    #print(inpath, lengths)
    for root, dirs, files in os.walk(inpath):
        #print("root: %s" % root)
    
        for name in files:        
            if (znum > (lz-1)):
                break
    
            image_name = os.path.join(root,name)
            image = None
            #print("image_name: %s" %image_name)        
            
            try: 
                image = pydicom.dcmread(image_name)
                #print(image_name)
                #seriesnumber = "%s" %image.SeriesNumber
                #imagetype = "%s" %image[0x2005,0x1011].value
                #InstanceNumber = "%s"%image.InstanceNumber
                SeriesInstanceUID = "%s" %image.SeriesInstanceUID
                SOPInstanceUID = "%s" %image.SOPInstanceUID
                UIDstart = SeriesInstanceUID[:29]
                SOPstart = SOPInstanceUID[:29]
                SeriesInstanceUID = UIDstart+modification_date+modification_time
                SOPInstanceUI = SOPstart+modification_date+modification_time + str(znum)
                image.SeriesNumber = 101 + number
                image.ProtocolName = protname
                image.SeriesInstanceUID = SeriesInstanceUID
                image.SOPInstanceUID = SOPInstanceUI
                image.SeriesDescription = protname
                tempimage = (indata[znum]).astype(np.uint16)
                image.PixelData = tempimage.tostring()
                znumout = znum + 1 + number*lz;
                if (znumout < 10):
                    outname = "IM_000" + str(znumout)
                elif (znumout < 100):
                    outname = "IM_00" + str(znumout)
                elif (znumout < 1000):
                    outname = "IM_0" + str(znumout)
                else:
                    outname = "IM_" + str(znumout)
                outfile = os.path.join(outpath,'%s' %outname)
                #print(outfile)
                pydicom.write_file(outfile, image)
                znum += 1
            except:
                # not a pydicom file, delete
                print("Ignoring: %s" % image_name)

# define data set class (for training)
class MyDataset(Dataset):
    def __init__(self, inpatch, outpatch):
        self.inpatch = inpatch
        self.outpatch = outpatch
        
    def __getitem__(self, index):
        x = self.inpatch[index]
        y = self.outpatch[index]
        
        return x, y
    
    def __len__(self):
        return len(self.inpatch)

# training function 1 - without learning rate decay (note: not used in main code)
def train_old(dataloader, model, loss_fn, optimizer):
    
    size = len(dataloader.dataset)
    train_loss = 0
    num_batches = len(dataloader)

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)
        train_loss += float(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        printstep = int(num_batches/10.0)
        if batch % printstep == 0:
            loss, current = float(loss), (batch + printstep) * len(x) 
            print(f"loss: {loss:>10.8f}  [{current:>5d}/{size:>5d}]")

    #scheduler.step()
    train_loss /= num_batches
    print(f"Training  : average loss {train_loss:>8f}")
    
    return train_loss

# training function 2 - with learning rate decay via scheduler
def train(dataloader, model, loss_fn, optimizer, scheduler):
    
    train_loss = 0
    num_batches = len(dataloader)

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred,y)
        train_loss += float(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        printstep = int(num_batches/10.0)
        if batch % printstep == 0:
            loss, current = float(loss), batch+1 
            print(f"loss: {loss:>10.8f}  [{current:>5d}/{num_batches:>5d}]")

    scheduler.step()
    train_loss /= num_batches
    print(f"Training  : average loss {train_loss:>8f}")
    
    return train_loss

# training function 2 - with learning rate decay via scheduler
def train2(dataloader, model, loss_fn, optimizer, scheduler):
    
    train_loss = 0
    num_batches = len(dataloader)

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss_pixels = w_pixels * loss_fn(pred,y)
        lssim       = w_ssim* ssimloss(pred,y)
        loss        = loss_pixels + lssim
        train_loss += float(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        printstep = int(num_batches/10.0)
        if batch % printstep == 0:
            tloss, fp, fs, current = float(loss), float(loss_pixels), float(lssim), batch+1 
            print(f"loss {tloss:>10.8f} > pixels: {fp:>10.8f} - ssim: {fs:>10.8f} [{current:>5d}/{num_batches:>5d}]")

    scheduler.step()
    train_loss /= num_batches
    print(f"Training  : average loss {train_loss:>8f}")
    
    return train_loss

# training function 2 - with learning rate decay via scheduler
def train3(dataloader, model, loss_fn, optimizer, scheduler):
    
    train_loss = 0
    num_batches = len(dataloader)

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss_pixels = w_pixels * loss_fn(pred,y)
        loss_edges  = w_edges  * edge_loss2(pred,y)
        lssim       = w_ssim   * ssimloss(pred,y)
        loss        = loss_pixels + lssim + loss_edges
        train_loss += float(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        printstep = int(num_batches/10.0)
        if batch % printstep == 0:
            tloss, fp, fs, fe, current = float(loss), float(loss_pixels), float(lssim), float(loss_edges), batch+1 
            print(f"loss {tloss:>10.8f} > pixels: {fp:>10.8f} - ssim: {fs:>10.8f} - edges: {fe:>10.8f} [{current:>5d}/{num_batches:>5d}]")

    scheduler.step()
    train_loss /= num_batches
    print(f"Training  : average loss {train_loss:>8f}")
    
    return train_loss


# validation function
def validate(dataloader, model, vloss_fn):
    
    num_batches = len(dataloader)
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_loss += float(vloss_fn(pred, y))

    val_loss /= num_batches
    print(f"Validation: average loss {val_loss:>8f} \n")
    
    return val_loss

# validation function
def validate2(dataloader, model, vloss_fn):
    
    num_batches = len(dataloader)
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss_pixels = vloss_fn(pred,y)
            lssim  = ssimloss(pred,y)
            loss = w_pixels * loss_pixels + w_ssim * lssim
            val_loss += float(loss)

    val_loss /= num_batches
    print(f"Validation: average loss {val_loss:>8f} \n")
    
    return val_loss

# validation function
def validate3(dataloader, model, vloss_fn):
    
    num_batches = len(dataloader)
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss_pixels = vloss_fn(pred,y)
            loss_edges  = edge_loss2(pred,y)
            lssim       = ssimloss(pred,y)
            loss        = w_pixels * loss_pixels + w_ssim * lssim + w_edges * loss_edges
            val_loss   += float(loss)

    val_loss /= num_batches
    print(f"Validation: average loss {val_loss:>8f} \n")
    
    return val_loss

# deploy on test set to calculate loss (note: not used in main code)
def deploy(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    average_loss = 0.0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)
        average_loss += float(loss)

        printstep = 1
        if batch % printstep == 0:
            loss, current = float(loss), (batch + printstep) * len(x)
            print(f"loss: {loss:>10.8f}  [{current:>5d}/{size:>5d}]")

    average_loss /= num_batches
    print(f"Average loss {average_loss:>8f} \n")

# define ResNet
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.convin = torch.nn.Conv2d(1, nrfilters, kernel_size=[kernel, kernel], padding='same', bias='True')
        self.conv1 = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[kernel, kernel], padding='same', bias='True')
        self.conv2 = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[kernel, kernel], padding='same', bias='True')
        self.conv3 = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[kernel, kernel], padding='same', bias='True')
        self.conv4 = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[kernel, kernel], padding='same', bias='True')
        self.conv5 = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[kernel, kernel], padding='same', bias='True')
        self.conv6 = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[kernel, kernel], padding='same', bias='True')
        self.conv7 = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[kernel, kernel], padding='same', bias='True')
        self.convout = torch.nn.Conv2d(nrfilters, 1, kernel_size=[kernel, kernel], padding='same', bias='True')
        self.relu = torch.nn.ReLU()
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x):
        y = self.relu(self.convin(x))
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))
        y = self.relu(self.conv3(y))
        y = self.relu(self.conv4(y))
        y = self.relu(self.conv5(y))
        y = self.relu(self.conv6(y))
        y = self.relu(self.conv7(y))
        y = self.convout(y)
        z = y+x
#       alternative for adding: z = torch.add(x,y) 
        return z

# Define small UResNet model (non-sequential)
class UResnetSmall(nn.Module):
    def __init__(self):
        super(UResnetSmall, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=[3, 3], padding='same', bias='True')
        self.conv2 = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv3 = torch.nn.Conv2d(2*nrfilters, nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv4 = torch.nn.Conv2d(nrfilters, 1, kernel_size=[3, 3], padding='same', bias='True')
        self.relu = torch.nn.ReLU()
        self.down = torch.nn.MaxPool2d(kernel_size=[2,2], stride=2)
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):

        y1 = self.relu(self.conv1(x))
        y2 = self.down(y1)
        y3 = self.relu(self.conv2(y2))
        y4 = self.up(y3)
        y5 = torch.cat((y4,y1),1)
        y6 = self.relu(self.conv3(y5))
        y7 = self.conv4(y6)
        out = x+y7
        
        return out

# Define larger UResNet model (non-sequential)
class UResnet2(nn.Module):
    def __init__(self):
        super(UResnet2, self).__init__()
        self.convin = torch.nn.Conv2d(1, nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv1d = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv2d = torch.nn.Conv2d(nrfilters, 2*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv3d = torch.nn.Conv2d(2*nrfilters, 2*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv4d = torch.nn.Conv2d(2*nrfilters, 4*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv5d = torch.nn.Conv2d(4*nrfilters, 4*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv6d = torch.nn.Conv2d(4*nrfilters, 8*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv7d = torch.nn.Conv2d(8*nrfilters, 8*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv8d = torch.nn.Conv2d(8*nrfilters, 16*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv1u = torch.nn.Conv2d(16*nrfilters, 8*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv2u = torch.nn.Conv2d(16*nrfilters, 8*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv3u = torch.nn.Conv2d(8*nrfilters, 8*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv4u = torch.nn.Conv2d(8*nrfilters, 4*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv5u = torch.nn.Conv2d(8*nrfilters, 4*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv6u = torch.nn.Conv2d(4*nrfilters, 4*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv7u = torch.nn.Conv2d(4*nrfilters, 2*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv8u = torch.nn.Conv2d(4*nrfilters, 2*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv9u = torch.nn.Conv2d(2*nrfilters, 2*nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv10u = torch.nn.Conv2d(2*nrfilters, nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv11u = torch.nn.Conv2d(2*nrfilters, nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.conv12u = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.convout = torch.nn.Conv2d(nrfilters, 1, kernel_size=[3, 3], padding='same', padding_mode = 'replicate', bias='True')
        self.relu = torch.nn.ReLU()
        self.down = torch.nn.MaxPool2d(kernel_size=[2,2], stride=2)
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

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
        y6 = torch.cat((y6,y4),1)
        y6 = self.relu(self.conv2u(y6))
        y6 = self.relu(self.conv3u(y6))
        y6 = self.up(y6)
        #
        y7 = self.relu(self.conv4u(y6))
        y7 = torch.cat((y7,y3),1)
        y7 = self.relu(self.conv5u(y7))
        y7 = self.relu(self.conv6u(y7))
        y7 = self.up(y7)
        #
        y8 = self.relu(self.conv7u(y7))
        y8 = torch.cat((y8,y2),1)
        y8 = self.relu(self.conv8u(y8))
        y8 = self.relu(self.conv9u(y8))
        y8 = self.up(y8)
        #        
        y9 = self.relu(self.conv10u(y8))
        y9 = torch.cat((y9,y1),1)
        y9 = self.relu(self.conv11u(y9))
        y9 = self.relu(self.conv12u(y9))
        #      
        y10 = self.convout(y9)
        #
        out = y10 + x
        
        return out
    
# Define larger UResNet model (non-sequential)
class UResnet(nn.Module):
    def __init__(self):
        super(UResnet, self).__init__()
        self.convin = torch.nn.Conv2d(1, nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv1d = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv2d = torch.nn.Conv2d(nrfilters, 2*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv3d = torch.nn.Conv2d(2*nrfilters, 2*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv4d = torch.nn.Conv2d(2*nrfilters, 4*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv5d = torch.nn.Conv2d(4*nrfilters, 4*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv6d = torch.nn.Conv2d(4*nrfilters, 8*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv7d = torch.nn.Conv2d(8*nrfilters, 8*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv8d = torch.nn.Conv2d(8*nrfilters, 16*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv1u = torch.nn.Conv2d(16*nrfilters, 8*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv2u = torch.nn.Conv2d(16*nrfilters, 8*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv3u = torch.nn.Conv2d(8*nrfilters, 8*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv4u = torch.nn.Conv2d(8*nrfilters, 4*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv5u = torch.nn.Conv2d(8*nrfilters, 4*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv6u = torch.nn.Conv2d(4*nrfilters, 4*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv7u = torch.nn.Conv2d(4*nrfilters, 2*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv8u = torch.nn.Conv2d(4*nrfilters, 2*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv9u = torch.nn.Conv2d(2*nrfilters, 2*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv10u = torch.nn.Conv2d(2*nrfilters, nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv11u = torch.nn.Conv2d(2*nrfilters, nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv12u = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.convout = torch.nn.Conv2d(nrfilters, 1, kernel_size=[3, 3], padding='same', bias='True')
        self.relu = torch.nn.ReLU()
        self.down = torch.nn.MaxPool2d(kernel_size=[2,2], stride=2)
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

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
        y6 = torch.cat((y6,y4),1)
        y6 = self.relu(self.conv2u(y6))
        y6 = self.relu(self.conv3u(y6))
        y6 = self.up(y6)
        #
        y7 = self.relu(self.conv4u(y6))
        y7 = torch.cat((y7,y3),1)
        y7 = self.relu(self.conv5u(y7))
        y7 = self.relu(self.conv6u(y7))
        y7 = self.up(y7)
        #
        y8 = self.relu(self.conv7u(y7))
        y8 = torch.cat((y8,y2),1)
        y8 = self.relu(self.conv8u(y8))
        y8 = self.relu(self.conv9u(y8))
        y8 = self.up(y8)
        #        
        y9 = self.relu(self.conv10u(y8))
        y9 = torch.cat((y9,y1),1)
        y9 = self.relu(self.conv11u(y9))
        y9 = self.relu(self.conv12u(y9))
        #      
        y10 = self.convout(y9)
        #
        out = y10 + x
        
        return out

# Define U-Net model (non-sequential)
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.convin = torch.nn.Conv2d(1, nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv1d = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv2d = torch.nn.Conv2d(nrfilters, 2*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv3d = torch.nn.Conv2d(2*nrfilters, 2*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv4d = torch.nn.Conv2d(2*nrfilters, 4*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv5d = torch.nn.Conv2d(4*nrfilters, 4*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv6d = torch.nn.Conv2d(4*nrfilters, 8*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv7d = torch.nn.Conv2d(8*nrfilters, 8*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv8d = torch.nn.Conv2d(8*nrfilters, 16*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv1u = torch.nn.Conv2d(16*nrfilters, 8*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv2u = torch.nn.Conv2d(16*nrfilters, 8*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv3u = torch.nn.Conv2d(8*nrfilters, 8*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv4u = torch.nn.Conv2d(8*nrfilters, 4*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv5u = torch.nn.Conv2d(8*nrfilters, 4*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv6u = torch.nn.Conv2d(4*nrfilters, 4*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv7u = torch.nn.Conv2d(4*nrfilters, 2*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv8u = torch.nn.Conv2d(4*nrfilters, 2*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv9u = torch.nn.Conv2d(2*nrfilters, 2*nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv10u = torch.nn.Conv2d(2*nrfilters, nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv11u = torch.nn.Conv2d(2*nrfilters, nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.conv12u = torch.nn.Conv2d(nrfilters, nrfilters, kernel_size=[3, 3], padding='same', bias='True')
        self.convout = torch.nn.Conv2d(nrfilters, 1, kernel_size=[3, 3], padding='same', bias='True')
        self.relu = torch.nn.ReLU()
        self.down = torch.nn.MaxPool2d(kernel_size=[2,2], stride=2)
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

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
        y6 = torch.cat((y6,y4),1)
        y6 = self.relu(self.conv2u(y6))
        y6 = self.relu(self.conv3u(y6))
        y6 = self.up(y6)
        #
        y7 = self.relu(self.conv4u(y6))
        y7 = torch.cat((y7,y3),1)
        y7 = self.relu(self.conv5u(y7))
        y7 = self.relu(self.conv6u(y7))
        y7 = self.up(y7)
        #
        y8 = self.relu(self.conv7u(y7))
        y8 = torch.cat((y8,y2),1)
        y8 = self.relu(self.conv8u(y8))
        y8 = self.relu(self.conv9u(y8))
        y8 = self.up(y8)
        #        
        y9 = self.relu(self.conv10u(y8))
        y9 = torch.cat((y9,y1),1)
        y9 = self.relu(self.conv11u(y9))
        y9 = self.relu(self.conv12u(y9))
        #      
        out = self.convout(y9)
        
        return out

# test dataset class definition (only input patches, no references assumed)
class MyTestset(Dataset):
    def __init__(self, inpatch):
        self.inpatch = inpatch
        
    def __getitem__(self, index):
        x = self.inpatch[index]
        
        return x
    
    def __len__(self):
        return len(self.inpatch)


# predict per batch using trained network
def predict_fast(dataloader, model):
    
    model.eval()
    
    tsize = len(dataloader.dataset)
    num_batches = len(dataloader)
    num_patches = num_batches * batch_size
    out = np.zeros((num_patches, 1, patch_size_x, patch_size_y), dtype=float)
    count = 0    
    batch = 0
    print('Size of dataset:', tsize, 'patches in', num_batches, 'batches of size', batch_size)
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            # pred = model(x)
            tmp = torch.Tensor.cpu(model(x))
            npred = tmp.detach().numpy()
            size = np.shape(npred)[0]
            out[count:count+size, 0,:,:] = npred[0:size,0,:,:]
            count = count+size
            batch = batch + 1
            printstep = int(num_batches/10.0)
            if batch % printstep == 0:
                print(f"--[{batch:>4d}/{num_batches:>5d}]")

    return out

# deploy per batch on complete training set 
def deploy_fast(dataloader, model, loss_fn):

    model.eval()

    tsize = len(dataloader.dataset)
    num_batches = len(dataloader)
    num_patches = num_batches * batch_size
    out = np.zeros((num_patches, 1, patch_size_x, patch_size_y), dtype=float)
    average_loss = 0.0
    count = 0
    print('Size of dataset:', tsize, 'patches in', num_batches, 'batches of size', batch_size)
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            tmp = torch.Tensor.cpu(pred)
            npred = tmp.detach().numpy()
            size = np.shape(npred)[0]
            out[count:count+size,0,:,:] = npred[0:size,0,:,:]
            count = count+size
            loss = loss_fn(pred, y)
            average_loss += float(loss)
            printstep = int(num_batches/10.0)
            if batch % printstep == 0:
                loss, current = float(loss), batch+1
                print(f"loss: {loss:>10.8f}  [{current:>5d}/{num_batches:>5d}]")

    average_loss /= num_batches
    print(f"Average loss {average_loss:>8f} \n")
    
    return out

# edge loss function (sobel)
def edge_loss(out, target):
    
#    px = target.size(2)
#    py = target.size(3)
#    npixels = px * py
    x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    convx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding='same', padding_mode = 'replicate', bias=False)
    convy = nn.Conv2d(1, 1, kernel_size=3 , stride=1, padding='same', padding_mode = 'replicate',bias=False)
    weights_x = torch.from_numpy(x_filter).float().unsqueeze(0).unsqueeze(0)
    weights_y = torch.from_numpy(y_filter).float().unsqueeze(0).unsqueeze(0)

    weights_x = weights_x.to(device)
    weights_y = weights_y.to(device)

    convx.weight = nn.Parameter(weights_x)
    convy.weight = nn.Parameter(weights_y)

    g1_x = convx(out)
    g2_x = convx(target)
    g1_y = convy(out)
    g2_y = convy(target)
    
    g_1 = torch.sqrt(g1_x * g1_x + g1_y * g1_y + 1e-8) # edge map prediction
    g_2 = torch.sqrt(g2_x * g2_x + g2_y * g2_y + 1e-8) # edge map target
    
    diff = g_1 - g_2
    diff2 = diff * diff
    
#   eloss = torch.mean(diff2)/float(npixels) # MSE of Sobel difference
    eloss = torch.mean(diff2) # MSE of Sobel edge difference
#    if (eloss > 1.0):
#        eloss = 1.0

    return eloss

# edge loss function 2 (sobel*mae)
def edge_loss3(out, target):

    x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    convx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding='same', padding_mode = 'replicate', bias=False)
    convy = nn.Conv2d(1, 1, kernel_size=3 , stride=1, padding='same', padding_mode = 'replicate',bias=False)
    weights_x = torch.from_numpy(x_filter).float().unsqueeze(0).unsqueeze(0)
    weights_y = torch.from_numpy(y_filter).float().unsqueeze(0).unsqueeze(0)

    weights_x = weights_x.to(device)
    weights_y = weights_y.to(device)

    convx.weight = nn.Parameter(weights_x)
    convy.weight = nn.Parameter(weights_y)

    g_x = convx(target) # gradient in x
    g_y = convy(target) # gradient in y
    
    g_tot = torch.sqrt(g_x * g_x + g_y * g_y + 1e-8) # Sobel edge map
    
    pixel_diff = torch.abs(out - target)   # absolute pixel difference
    tloss = torch.mean(g_tot * pixel_diff) # mean of (pixel difference * edge_strength)

    return tloss

# edge loss function 2 (sobel*mae)
def edge_loss2(out, target):

    x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    convx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding='same', padding_mode = 'replicate', bias=False)
    convy = nn.Conv2d(1, 1, kernel_size=3 , stride=1, padding='same', padding_mode = 'replicate',bias=False)
    weights_x = torch.from_numpy(x_filter).float().unsqueeze(0).unsqueeze(0)
    weights_y = torch.from_numpy(y_filter).float().unsqueeze(0).unsqueeze(0)

    weights_x = weights_x.to(device)
    weights_y = weights_y.to(device)

    convx.weight = nn.Parameter(weights_x)
    convy.weight = nn.Parameter(weights_y)

    g_x = convx(target) # gradient in x
    g_y = convy(target) # gradient in y
    
    g_tot = torch.sqrt(g_x * g_x + g_y * g_y + 1e-8) # Sobel edge map
    
    pixel_diff = torch.abs(out - target)    # absolute pixel difference
    max_diff = float(torch.max(pixel_diff)) # max error
    tloss = max_diff * torch.mean(g_tot)    # max error * edge_strength

    return tloss

# -------------------------------------------------------------------------------------------

# patch size and overlap parameters
patchSize    = np.array([16,16])
patch_size_x = patchSize[0]
patch_size_y = patchSize[1]
ovlPerc      = np.array([0.5, 0.5])

# training parameters
dataAugmentation  = True
nrfilters         = 32
kernel            = 3
batch_size        = 512
nrEpochs = epochs = int(500)
min_nr_epochs     = 10
patience          = 25
perc_validation   = 0.1
learning_rate     = 0.01
gamma_scheduler   = 0.99999

# loss function weightings
w_pixels          = 1.0
w_edges           = 0.1
w_ssim            = 0.0

# file parameters
exportImages = True
nrExport     = 83
protocolName = "2D UResNet pytorch 83 (el2) DA"
model_file   = "D:\\MR-SR\\models\\bestModelSoFar.pt"

# get the data
seriesnumberselected1 = "1001"
seriesnumberselected2 = "801"
seriesnumberselected3 = "1301"
seriesnumberselected4 = "201"

try:
    # training data
    in_path1 = 'D:\\MR-SR\\trainingdata\\vol4Dixon\\' + seriesnumberselected1
    out_path1 = 'D:\\MR-SR\\trainingdata\\vol4Dixon\\' + seriesnumberselected1 + '\\DICOM'
    in_path2 = 'D:\\MR-SR\\trainingdata\\vol3Dixon\\' + seriesnumberselected2
    out_path2 = 'D:\\MR-SR\\trainingdata\\vol3Dixon\\' + seriesnumberselected2 + '\\DICOM'
    in_path3 = 'D:\\MR-SR\\trainingdata\\vol2Dixon\\' + seriesnumberselected3
    out_path3 = 'D:\\MR-SR\\trainingdata\\vol2Dixon\\' + seriesnumberselected3 + '\\DICOM'
    in_path_training = 'D:\\MR-SR\\trainingdata'
    # test data    
    in_path_test1 = 'D:\\MR-SR\\trainingdata\\vol4Dixon\\' + seriesnumberselected4
    out_path_test1 = 'D:\\MR-SR\\trainingdata\\vol4Dixon\\' + seriesnumberselected4 + '\\DICOM'    
    in_path_test2 = 'D:\\MR-SR\\testdata'
    out_path_test2 = 'D:\\MR-SR\\testdata\\401\\DICOM'
except:
    print("Need input and output paths")
    sys.exit(1)

# load the training and test data
infile1 = in_path1 + "\\TB15_20190829_vol4dixon_" + seriesnumberselected1 + "_res252_3d.npz"
infile2 = in_path2 + "\\TB15_20190829_vol3dixon_" + seriesnumberselected2 + "_res252_3d.npz"
infile3 = in_path3 + "\\TB15_20190829_vol2dixon_" + seriesnumberselected3 + "_res252_3d.npz"
infile4 = in_path_training + "\\TB16_20190930_vol2_707_x220.npz"
infile5 = in_path_training + "\\TB16_20191125_vol1_301_x220.npz"
infile6 = in_path_training + "\\TB16_20191125_vol2_501_x220.npz"
infile7 = in_path_training + "\\TB16_20191125_vol1_801_x220.npz"
infile8 = in_path_training + "\\TB16_20191125_vol2_201_x220.npz"
infiletest1 = in_path_test1 + "\\TB15_20190829_vol4dixon_" + seriesnumberselected4 + "_res252_3d.npz"
infiletest2 = in_path_test2 + "\\TB16_20191125_401.npz"
npzfile1 = np.load(infile1)
npzfile2 = np.load(infile2)
npzfile3 = np.load(infile3)
npzfile4 = np.load(infile4)
npzfile5 = np.load(infile5)
npzfile6 = np.load(infile6)
npzfile7 = np.load(infile7)
npzfile8 = np.load(infile8)
npzfiletest1 = np.load(infiletest1)
npzfiletest2 = np.load(infiletest2)
lowRes1  = npzfile1['lr']
highRes1 = npzfile1['hr']
lowRes2  = npzfile2['lr']
highRes2 = npzfile2['hr']
lowRes3  = npzfile3['lr']
highRes3 = npzfile3['hr']
lowRes4  = npzfile4['lr']
highRes4 = npzfile4['hr']
lowRes5  = npzfile5['lr']
highRes5 = npzfile5['hr']
lowRes6  = npzfile6['lr']
highRes6 = npzfile6['hr']
lowRes7  = npzfile7['lr']
highRes7 = npzfile7['hr']
lowRes8  = npzfile8['lr']
highRes8 = npzfile8['hr']
lowResTest1  = npzfiletest1['lr']
lowResTest2  = npzfiletest2['lr']

print('Matrix of imported images:')
print('Train 1: ', np.shape(lowRes1), np.shape(highRes1))
print('Train 2: ', np.shape(lowRes2), np.shape(highRes2))
print('Train 3: ', np.shape(lowRes3), np.shape(highRes3))
print('Train 4: ', np.shape(lowRes4), np.shape(highRes4))
print('Train 5: ', np.shape(lowRes5), np.shape(highRes5))
print('Train 6: ', np.shape(lowRes6), np.shape(highRes6))
print('Train 7: ', np.shape(lowRes7), np.shape(highRes7))
print('Train 8: ', np.shape(lowRes8), np.shape(highRes8))
print('Test  1: ', np.shape(lowResTest1))
print('Test  2: ', np.shape(lowResTest2))

# select VOIs containing relevant data
RFOV1 = 4
RFOV2 = 6
input_size = np.array(lowRes1.shape)
select_size = (np.array(input_size)).astype(np.int)
select_size[1] *= RFOV1/9
select_size[2] *= RFOV2/9

xl = select_size[1]
xs = int((input_size[1]-xl)/2) + 20
xs1 = int((input_size[1]-xl)/2)
yl = select_size[2]
ys = int((input_size[2]-yl)/2)
zl = input_size[0]
zs = 10
ze = 156
nrSlices = ze - zs

lowResSelect1 = (np.absolute(lowRes1[zs:ze,xs:xs+xl,ys:ys+yl])).astype(np.float32)
highResSelect1 = (np.absolute(highRes1[zs:ze,xs:xs+xl,ys:ys+yl])).astype(np.float32)
lowResSelect2 = (np.absolute(lowRes2[zs:ze,xs:xs+xl,ys:ys+yl])).astype(np.float32)
highResSelect2 = (np.absolute(highRes2[zs:ze,xs:xs+xl,ys:ys+yl])).astype(np.float32)
lowResSelect3 = (np.absolute(lowRes3[zs:ze,xs:xs+xl,ys:ys+yl])).astype(np.float32)
highResSelect3 = (np.absolute(highRes3[zs:ze,xs:xs+xl,ys:ys+yl])).astype(np.float32)
lowResSelect4 = (np.absolute(lowRes4[zs:ze,xs:xs+xl,ys:ys+yl])).astype(np.float32)
highResSelect4 = (np.absolute(highRes4[zs:ze,xs:xs+xl,ys:ys+yl])).astype(np.float32)
lowResSelect5 = (np.absolute(lowRes5[zs:ze,xs1:xs1+xl,ys:ys+yl])).astype(np.float32)
highResSelect5 = (np.absolute(highRes5[zs:ze,xs1:xs1+xl,ys:ys+yl])).astype(np.float32)
lowResSelect6 = (np.absolute(lowRes6[zs:ze,xs1:xs1+xl,ys:ys+yl])).astype(np.float32)
highResSelect6 = (np.absolute(highRes6[zs:ze,xs1:xs1+xl,ys:ys+yl])).astype(np.float32)
lowResSelect7 = (np.absolute(lowRes7[zs:ze,xs1:xs1+xl,ys:ys+yl])).astype(np.float32)
highResSelect7 = (np.absolute(highRes7[zs:ze,xs1:xs1+xl,ys:ys+yl])).astype(np.float32)
lowResSelect8 = (np.absolute(lowRes8[zs:ze,xs1:xs1+xl,ys:ys+yl])).astype(np.float32)
highResSelect8 = (np.absolute(highRes8[zs:ze,xs1:xs1+xl,ys:ys+yl])).astype(np.float32)
lowResSelectTest1 = (np.absolute(lowResTest1)).astype(np.float32)
lowResSelectTest2 = (np.absolute(lowResTest2)).astype(np.float32)

# determine 95% values (as estimate of the maximum value)
maxValueHR1 = np.percentile(highResSelect1,95)
maxValueHR2 = np.percentile(highResSelect2,95)
maxValueHR3 = np.percentile(highResSelect3,95)
maxValueHR4 = np.percentile(highResSelect4,95)
maxValueHR5 = np.percentile(highResSelect5,95)
maxValueHR6 = np.percentile(highResSelect6,95)
maxValueHR7 = np.percentile(highResSelect7,95)
maxValueHR8 = np.percentile(highResSelect8,95)
maxValueLRTest1 = np.percentile(lowResSelectTest1,95)
maxValueLRTest2 = np.percentile(lowResSelectTest2,95)

# print 95% values
print(1,np.amax(lowResSelect1),np.amax(highResSelect1))
print(2,np.amax(lowResSelect2),np.amax(highResSelect2))
print(3,np.amax(lowResSelect3),np.amax(highResSelect3))
print(4,np.amax(lowResSelect4),np.amax(highResSelect4))
print(5,np.amax(lowResSelect5),np.amax(highResSelect5))
print(6,np.amax(lowResSelect6),np.amax(highResSelect6))
print(7,np.amax(lowResSelect7),np.amax(highResSelect7))
print(8,np.amax(lowResSelect8),np.amax(highResSelect8))

# scale the training and test data, map 95% value to 1.0
lowResSelect1  = lowResSelect1/maxValueHR1
highResSelect1 = highResSelect1/maxValueHR1
lowResSelect2  = lowResSelect2/maxValueHR2
highResSelect2 = highResSelect2/maxValueHR2
lowResSelect3  = lowResSelect3/maxValueHR3
highResSelect3 = highResSelect3/maxValueHR3
lowResSelect4  = lowResSelect4/maxValueHR4
highResSelect4 = highResSelect4/maxValueHR4
lowResSelect5  = lowResSelect5/maxValueHR5
highResSelect5 = highResSelect5/maxValueHR5
lowResSelect6  = lowResSelect6/maxValueHR6
highResSelect6 = highResSelect6/maxValueHR6
lowResSelect7  = lowResSelect7/maxValueHR7
highResSelect7 = highResSelect7/maxValueHR7
lowResSelect8  = lowResSelect8/maxValueHR8
highResSelect8 = highResSelect8/maxValueHR8
lowResSelectTest1  = lowResSelectTest1/maxValueLRTest1
lowResSelectTest2  = lowResSelectTest2/maxValueLRTest2

# print max values
print('Max values training data:')
print('Scan 1 max lowres and highres: ', np.amax(lowResSelect1),np.amax(highResSelect1))
print('Scan 2 max lowres and highres: ', np.amax(lowResSelect2),np.amax(highResSelect2))
print('Scan 3 max lowres and highres: ', np.amax(lowResSelect3),np.amax(highResSelect3))
print('Scan 4 max lowres and highres: ', np.amax(lowResSelect4),np.amax(highResSelect4))
print('Scan 5 max lowres and highres: ', np.amax(lowResSelect5),np.amax(highResSelect5))
print('Scan 6 max lowres and highres: ', np.amax(lowResSelect6),np.amax(highResSelect6))
print('Scan 7 max lowres and highres: ', np.amax(lowResSelect7),np.amax(highResSelect7))
print('Scan 8 max lowres and highres: ', np.amax(lowResSelect8),np.amax(highResSelect8))
print('Max values test data:')
print('Scan 1:', maxValueLRTest1)
print('Scan 2:', maxValueLRTest2)

# construct difference images
dif1 = highResSelect1-lowResSelect1
dif2 = highResSelect2-lowResSelect2
dif3 = highResSelect3-lowResSelect3
dif4 = highResSelect4-lowResSelect4
dif5 = highResSelect5-lowResSelect5
dif6 = highResSelect6-lowResSelect6
dif7 = highResSelect7-lowResSelect7
dif8 = highResSelect8-lowResSelect8

# plot lowres, highres and difference images
plt.figure(figsize=(16, 16))
img_index = 0
ax = plt.subplot(8,3,1)
img1 = np.squeeze(lowResSelect1[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,2)
img1 = np.squeeze(highResSelect1[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,3)
img1 = np.squeeze(dif1[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,4)
img1 = np.squeeze(lowResSelect2[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,5)
img1 = np.squeeze(highResSelect2[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,6)
img1 = np.squeeze(dif2[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,7)
img1 = np.squeeze(lowResSelect3[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,8)
img1 = np.squeeze(highResSelect3[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,9)
img1 = np.squeeze(dif3[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,10)
img1 = np.squeeze(lowResSelect4[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,11)
img1 = np.squeeze(highResSelect4[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,12)
img1 = np.squeeze(dif4[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,13)
img1 = np.squeeze(lowResSelect5[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,14)
img1 = np.squeeze(highResSelect5[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,15)
img1 = np.squeeze(dif5[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,16)
img1 = np.squeeze(lowResSelect6[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,17)
img1 = np.squeeze(highResSelect6[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,18)
img1 = np.squeeze(dif6[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,19)
img1 = np.squeeze(lowResSelect7[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,20)
img1 = np.squeeze(highResSelect7[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,21)
img1 = np.squeeze(dif7[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,22)
img1 = np.squeeze(lowResSelect8[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,23)
img1 = np.squeeze(highResSelect8[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax = plt.subplot(8,3,24)
img1 = np.squeeze(dif8[img_index])
plt.imshow(img1, cmap=plt.cm.gray)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
plt.tight_layout()
plt.show()

# free memory (difference images no longer needed)
dif1 = None
dif2 = None
dif3 = None
dif4 = None
dif5 = None
dif6 = None
dif7 = None
dif8 = None

# create patches
size03D = (np.array(lowResSelect1.shape)).astype(np.int)
size0 = np.array([size03D[1], size03D[2]])
numberOfPatches = (np.divide(size0,(ovlPerc*patchSize))).astype(int) - 1
roundedFull = patchSize + (numberOfPatches - 1)*ovlPerc*patchSize
residual = size0 - roundedFull
deltaPix = np.divide(residual,(numberOfPatches - 1))
startArrayY = []
startArrayX = []
for x in range(numberOfPatches[0]):
    startArrayX.append(int(ovlPerc[0]*patchSize[1]*x + deltaPix[0]*x + 0.5))
for y in range(numberOfPatches[1]):
    startArrayY.append(int(ovlPerc[1]*patchSize[1]*y + deltaPix[1]*y + 0.5))
    
# print patch variables of training data
print('Training data:')
print('3D image size', size03D)
print('Patch size (x,y)', patchSize)
print('Nr. of patches (x,y)', numberOfPatches)
print('Delta pix (x,y)', deltaPix)

sizeTest3D = (np.array(lowResSelectTest1.shape)).astype(np.int)
sizeTest = np.array([sizeTest3D[1], sizeTest3D[2]])
numberOfPatchesTest = (np.divide(sizeTest,(ovlPerc*patchSize))).astype(int) - 1
roundedFullTest = patchSize + (numberOfPatchesTest - 1)*ovlPerc*patchSize
residualTest = sizeTest - roundedFullTest
deltaPixTest = np.divide(residualTest,(numberOfPatchesTest - 1))
startArrayYTest = []
startArrayXTest = []
for x in range(numberOfPatchesTest[0]):
    startArrayXTest.append(int(ovlPerc[0]*patchSize[0]*x + deltaPixTest[0]*x + 0.5))
for y in range(numberOfPatchesTest[1]):
    startArrayYTest.append(int(ovlPerc[1]*patchSize[1]*y + deltaPixTest[1]*y + 0.5))

# print patch variables of test data
nrSlicesTest = sizeTest3D[0]
print('Test data:')
print('3D image size', sizeTest3D)
print('Patch size (x,y)', patchSize)
print('Nr. of patches (x,y)', numberOfPatchesTest)
print('Delta pix (x,y)', deltaPixTest)

totalPatches = int(numberOfPatches[0]*numberOfPatches[1]*nrSlices)
outPutSize = (totalPatches*8,patchSize[0],patchSize[1]) # 8 training scans
totalPatchesTest = int(numberOfPatchesTest[0]*numberOfPatchesTest[1]*nrSlicesTest) 
outPutSizeTest = (totalPatchesTest*2,patchSize[0],patchSize[1]) # 2 test scans
reorderArrayLowRes = np.zeros(outPutSize, dtype = np.float32)
reorderArrayLowResTest1 = np.zeros(outPutSizeTest, dtype = np.float32)
reorderArrayLowResTest2 = np.zeros(outPutSizeTest, dtype = np.float32)
reorderArrayHighRes = np.zeros(outPutSize, dtype = np.float32)
count = 0
inOutMap = []
for zi in range(nrSlices):
    for xi in range(numberOfPatches[0]):
        startX = startArrayX[xi]
        endX = startX + patchSize[0]
        for yi in range(numberOfPatches[1]):
            startY = startArrayY[yi]
            endY = startY + patchSize[1]
            count2 = count + totalPatches
            count3 = count2 + totalPatches
            count4 = count3 + totalPatches
            count5 = count4 + totalPatches
            count6 = count5 + totalPatches
            count7 = count6 + totalPatches
            count8 = count7 + totalPatches
            reorderArrayLowRes[count][0:patchSize[0]][0:patchSize[1]] = lowResSelect1[zi,startX:endX,startY:endY]
            reorderArrayHighRes[count][0:patchSize[0]][0:patchSize[1]] = highResSelect1[zi,startX:endX,startY:endY]
            reorderArrayLowRes[count2][0:patchSize[0]][0:patchSize[1]] = lowResSelect2[zi,startX:endX,startY:endY]
            reorderArrayHighRes[count2][0:patchSize[0]][0:patchSize[1]] = highResSelect2[zi,startX:endX,startY:endY]
            reorderArrayLowRes[count3][0:patchSize[0]][0:patchSize[1]] = lowResSelect3[zi,startX:endX,startY:endY]
            reorderArrayHighRes[count3][0:patchSize[0]][0:patchSize[1]] = highResSelect3[zi,startX:endX,startY:endY]
            reorderArrayLowRes[count4][0:patchSize[0]][0:patchSize[1]] = lowResSelect4[zi,startX:endX,startY:endY]
            reorderArrayHighRes[count4][0:patchSize[0]][0:patchSize[1]] = highResSelect4[zi,startX:endX,startY:endY]
            reorderArrayLowRes[count5][0:patchSize[0]][0:patchSize[1]] = lowResSelect5[zi,startX:endX,startY:endY]
            reorderArrayHighRes[count5][0:patchSize[0]][0:patchSize[1]] = highResSelect5[zi,startX:endX,startY:endY]
            reorderArrayLowRes[count6][0:patchSize[0]][0:patchSize[1]] = lowResSelect6[zi,startX:endX,startY:endY]
            reorderArrayHighRes[count6][0:patchSize[0]][0:patchSize[1]] = highResSelect6[zi,startX:endX,startY:endY]
            reorderArrayLowRes[count7][0:patchSize[0]][0:patchSize[1]] = lowResSelect7[zi,startX:endX,startY:endY]
            reorderArrayHighRes[count7][0:patchSize[0]][0:patchSize[1]] = highResSelect7[zi,startX:endX,startY:endY]
            reorderArrayLowRes[count8][0:patchSize[0]][0:patchSize[1]] = lowResSelect8[zi,startX:endX,startY:endY]
            reorderArrayHighRes[count8][0:patchSize[0]][0:patchSize[1]] = highResSelect8[zi,startX:endX,startY:endY]
            count +=1

nrTrainingsPatches = totalPatches * 8

count = 0
for zi in range(nrSlicesTest):
    for xi in range(numberOfPatchesTest[0]):
        startX = startArrayXTest[xi]
        endX = startX + patchSize[0]
        for yi in range(numberOfPatchesTest[1]):
            startY = startArrayYTest[yi]
            endY = startY + patchSize[1]
            reorderArrayLowResTest1[count][0:patchSize[0]][0:patchSize[1]] = lowResSelectTest1[zi,startX:endX,startY:endY]
            reorderArrayLowResTest2[count][0:patchSize[0]][0:patchSize[1]] = lowResSelectTest2[zi,startX:endX,startY:endY]
            count +=1

# free memory
lowResSelect2  = None
highResSelect2 = None  
lowResSelect3  = None
highResSelect3 = None
lowResSelect4  = None
highResSelect4 = None 
lowResSelect5  = None
highResSelect5 = None
lowResSelect6  = None
highResSelect6 = None
lowResSelect7  = None
highResSelect7 = None
lowResSelect8  = None
highResSelect8 = None 

lowres_train = reorderArrayLowRes
highres_train = reorderArrayHighRes

# data augmentaiton (if requested)
newNumberOfPatches = numberOfPatches
newNrTrainingsPatches = nrTrainingsPatches
if dataAugmentation:

    # perform data augmentation, lr = left-right, ud = up-down in x,y; array[nr_patches, z, x, y] maps to indices 0, 1, 2 and 3 
    print('... data augmentation: flip x,y up-down and left-right')
    reorderArrayLowRes_lr = np.flip(reorderArrayLowRes, 1)
    reorderArrayLowRes_ud = np.flip(reorderArrayLowRes, 2)
    reorderArrayLowRes_tot = np.concatenate((reorderArrayLowRes, reorderArrayLowRes_lr, reorderArrayLowRes_ud))

    reorderArrayHighRes_lr = np.flip(reorderArrayHighRes, 1)
    reorderArrayHighRes_ud = np.flip(reorderArrayHighRes, 2)
    reorderArrayHighRes_tot = np.concatenate((reorderArrayHighRes, reorderArrayHighRes_lr, reorderArrayHighRes_ud))

    # free memory
    reorderArrayLowRes_lr = None
    reorderArrayLowRes_ud = None
    reorderArrayHighRes_lr = None
    reorderArrayHighRes_ud = None

    # perform data augmentation, rotate 90, 180 and 270 degrees in x,y
    print('... data augmentation: rotate x,y over 90, 180 and 270 degrees')
    reorderArrayLowRes_rot90 = np.rot90(reorderArrayLowRes, k=1, axes=(1,2))
    reorderArrayLowRes_rot180 = np.rot90(reorderArrayLowRes, k=2, axes=(1,2))
    reorderArrayLowRes_rot270 = np.rot90(reorderArrayLowRes, k=3, axes=(1,2))
    lowres_train = np.concatenate((reorderArrayLowRes_tot, reorderArrayLowRes_rot90, reorderArrayLowRes_rot180, reorderArrayLowRes_rot270))

    reorderArrayHighRes_rot90 = np.rot90(reorderArrayHighRes, k=1, axes=(1,2))
    reorderArrayHighRes_rot180 = np.rot90(reorderArrayHighRes, k=2, axes=(1,2))
    reorderArrayHighRes_rot270 = np.rot90(reorderArrayHighRes, k=3, axes=(1,2))
    highres_train = np.concatenate((reorderArrayHighRes_tot, reorderArrayHighRes_rot90, reorderArrayHighRes_rot180, reorderArrayHighRes_rot270))

    # total number of patches increased by factor 6
    newNumberOfPatches = 6*numberOfPatches
    newNrTrainingsPatches = 6*newNrTrainingsPatches
    print('Nr. patches per slice before data augmentation:', numberOfPatches)
    print('Nr. patches per slice after data augmentation:', newNumberOfPatches)

    # free memory
    reorderArrayLowRes_rot90 = None
    reorderArrayLowRes_rot180 = None
    reorderArrayLowRes_rot270 = None
    reorderArrayHighRes_rot90 = None
    reorderArrayHighRes_rot180 = None
    reorderArrayHighRes_rot270 = None
    reorderArrayLowRes_tot = None
    reorderArrayHighRes_tot = None

# divide in training and validation data
nval   = int(perc_validation * newNrTrainingsPatches) # for validation
ntrain = newNrTrainingsPatches - nval               # for training

# make sure the nr. of training and validation patches are both a multiple of the batch size
nval   = int(nval/batch_size) * batch_size
ntrain = int(ntrain/batch_size) * batch_size

# get data
lowres_tmp       = lowres_train
highres_tmp      = highres_train
lowres_validate  = lowres_tmp[:nval]
highres_validate = highres_tmp[:nval]
lowres_train     = lowres_tmp[nval:nval+ntrain]
highres_train    = highres_tmp[nval:nval+ntrain]
print('Nr. patches training:', ntrain, 'Nr. patches validation:', nval)

# format conversion for Pytorch
lowres_tmp       = np.expand_dims(lowres_tmp, axis=1)
highres_tmp      = np.expand_dims(highres_tmp, axis=1)
lowres_train     = np.expand_dims(lowres_train, axis=1)
highres_train    = np.expand_dims(highres_train, axis=1)
lowres_validate  = np.expand_dims(lowres_validate, axis=1)
highres_validate = np.expand_dims(highres_validate, axis=1)
lowres_test1     = np.expand_dims(reorderArrayLowResTest1, axis=1)
lowres_test2     = np.expand_dims(reorderArrayLowResTest2, axis=1)

# scale the complete training set
stdev            = np.std(highres_tmp)
scaleFactor      = (1.0/stdev)
lowres_train     = scaleFactor * lowres_train
highres_train    = scaleFactor * highres_train
lowres_validate  = scaleFactor * lowres_validate
highres_validate = scaleFactor * highres_validate
lowres_test1     = scaleFactor * lowres_test1
lowres_test2     = scaleFactor * lowres_test2
lowres_tmp       = scaleFactor * lowres_tmp
highres_tmp      = scaleFactor * highres_tmp
print('st dev highres train/validate = ', stdev, 'scaleFactor = ', scaleFactor, '\n')

#---------------
# train network
#---------------

# configure data loader
train_dataset    = MyDataset(lowres_train, highres_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataset      = MyDataset(lowres_validate, highres_validate)
val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# select model
myNetwork = UResnet2().to(device)
print(myNetwork)

# set loss function
#vloss_fn  = nn.MSELoss().to(device)
#loss_fn   = nn.MSELoss().to(device)
vloss_fn  = nn.L1Loss().to(device)
loss_fn   = nn.L1Loss().to(device)
optimizer = torch.optim.SGD(myNetwork.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_scheduler)

# speed optimization
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True

# perform actual training & validation
print("\nTraining & validation:")
start_time   = time.time()
train_loss_c = np.zeros(epochs, dtype='float')
val_loss_c   = np.zeros(epochs, dtype='float')
x_v          = np.zeros(epochs, dtype='int')
min_val_loss = np.Inf
true_epochs  = 0
size         = len(train_dataloader.dataset)
num_batches  = len(train_dataloader)
epochs_no_improve = 0
print('Size of training dataset:', size, 'patches in', num_batches, 'batches of size', batch_size, '\n')
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss_c[t] = train3(train_dataloader, myNetwork, loss_fn, optimizer, scheduler)
    val_loss = val_loss_c[t] = validate3(val_dataloader, myNetwork, vloss_fn)
    x_v[t] = t+1
    true_epochs = true_epochs + 1
    if val_loss < min_val_loss:
        torch.save(myNetwork, model_file)
        epochs_no_improve = 0
        min_val_loss = val_loss
        best_epoch = t+1
    else:
        epochs_no_improve += 1
        print('Validation loss did not improve in last', epochs_no_improve, 'epochs\n')
    if t > min_nr_epochs and epochs_no_improve == patience:
        print('Early stopping at epoch', t+1, '- best epoch is', best_epoch, 'with minimum loss', min_val_loss, '\n')
        early_stop = True
        break
    else:
        continue  

# copy relevant part of loss curves (for plotting)
end_time                   = time.time()
epochs                     = true_epochs
train_loss_curve           = np.zeros(epochs, dtype='float')
val_loss_curve             = np.zeros(epochs, dtype='float')
x_value                    = np.zeros(epochs, dtype='int')
train_loss_curve[0:epochs] = train_loss_c[0:epochs]
val_loss_curve[0:epochs]   = val_loss_c[0:epochs]
x_value[0:epochs]          = x_v[0:epochs]

# calculate training time in hours and minutes
thours   = (end_time - start_time)/3600.0
ihours   = int(thours)
tminutes = (thours - ihours)*60.0
iminutes = int(tminutes)
tsec     = (tminutes-iminutes)*60.0
isec     = int(tsec)

# save best model with best epoch number in filename
bestModel = torch.load(model_file)
bestModel_file = 'D:\\MR-SR\\Models\\myBestModel_at_epoch_'+str(best_epoch)+'.pt'
torch.save(bestModel, bestModel_file)
print("Training & validation done!")
print('Training took', ihours, 'hours,', iminutes, 'minutes and', isec, 'seconds')
print('Best model saved to:', bestModel_file, '\n')

# free memory (training and validation data no longer needed)
highres_train    = None
highres_validate = None
lowres_train     = None
lowres_validate  = None

# clear no longer needed memory
del vloss_fn
del train_dataset
del train_dataloader
del val_dataset
del val_dataloader
del myNetwork

# clear GPU cache
torch.cuda.empty_cache()

# use best model
myNetwork = bestModel.to(device)

# Calculate predicted highres images (on lowres training images)
print("Performance on all training & validation data:")
tindata = MyDataset(lowres_tmp, highres_tmp)
dataloader = DataLoader(tindata, batch_size=batch_size, shuffle=False)
highres_predicted = deploy_fast(dataloader, myNetwork, loss_fn)
highres_tmp = None
lowres_tmp = None

# clear memory
del tindata
del dataloader
torch.cuda.empty_cache()

# Calculate predicted highres images (on lowres test images)
print("Creating high-res test data 1:")
# prepare data loader
tindata = torch.Tensor(lowres_test1)
dataset = MyTestset(tindata)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
highres_predicted_test1 = predict_fast(dataloader, myNetwork)

# clear memory
del tindata
del dataloader
torch.cuda.empty_cache()

# Calculate predicted highres images (on lowres test images)
print("Creating high-res test data 2:")
tindata = torch.Tensor(lowres_test2)
dataset = MyTestset(tindata)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
highres_predicted_test2 = predict_fast(dataloader, myNetwork)

# clear memory
del tindata
del dataloader
del loss_fn
torch.cuda.empty_cache()

# convert back from tensors to numpy arrays and squeeze empty dimensions
highres_predicted = stdev * np.squeeze(highres_predicted)
highres_predicted_test1 = stdev * np.squeeze(highres_predicted_test1)
highres_predicted_test2 = stdev * np.squeeze(highres_predicted_test2)

# merge patches back into 3D images
highResRec1 = np.zeros(size03D, dtype = np.float32)
highResRec2 = np.zeros(size03D, dtype = np.float32)
highResRec3 = np.zeros(size03D, dtype = np.float32)
sumMap  = np.zeros(size03D, dtype = np.float32)
sumMap0 = np.zeros(patchSize, dtype = np.float32) + 1
count = 0
for zi in range(nrSlices):
    for xi in range(numberOfPatches[0]):
        startX = startArrayX[xi]
        endX = startX + patchSize[0]
        for yi in range(numberOfPatches[1]):
            startY = startArrayY[yi]
            endY = startY +  patchSize[1]
            count2 = count + totalPatches
            count3 = count2 + totalPatches
            sumMap[zi, startX:endX,startY:endY] += sumMap0
            highResRec1[zi,startX:endX,startY:endY] += highres_predicted[count][0:patchSize[0]][0:patchSize[1]]
            highResRec2[zi,startX:endX,startY:endY] += highres_predicted[count2][0:patchSize[0]][0:patchSize[1]]
            highResRec3[zi,startX:endX,startY:endY] += highres_predicted[count3][0:patchSize[0]][0:patchSize[1]]
            count +=1                

highResRecTest1 = np.zeros(sizeTest3D, dtype = np.float32)
highResRecTest2 = np.zeros(sizeTest3D, dtype = np.float32)
sumMapTest  = np.zeros(sizeTest3D, dtype = np.float32)
sumMap0 = np.zeros(patchSize, dtype = np.float32) + 1
count = 0
for zi in range(nrSlicesTest):
    for xi in range(numberOfPatchesTest[0]):
        startX = startArrayXTest[xi]
        endX = startX + patchSize[0]
        for yi in range(numberOfPatchesTest[1]):
            startY = startArrayYTest[yi]
            endY = startY + patchSize[1]
            sumMapTest[zi, startX:endX,startY:endY] += sumMap0
            highResRecTest1[zi,startX:endX,startY:endY] += highres_predicted_test1[count][0:patchSize[0]][0:patchSize[1]]
            highResRecTest2[zi,startX:endX,startY:endY] += highres_predicted_test2[count][0:patchSize[0]][0:patchSize[1]]
            count +=1                

highResRec1 = highResRec1*maxValueHR1
highResRec2 = highResRec2*maxValueHR2
highResRec3 = highResRec3*maxValueHR3
highResRecTest1 = highResRecTest1*maxValueLRTest1
highResRecTest2 = highResRecTest2*maxValueLRTest2

normHighResRec1 = np.divide(highResRec1,sumMap)
normHighResRec2 = np.divide(highResRec2,sumMap)
normHighResRec3 = np.divide(highResRec3,sumMap)
normHighResRecTest1 = np.divide(highResRecTest1,sumMapTest)
normHighResRecTest2 = np.divide(highResRecTest2,sumMapTest)

# export images
if exportImages:
    outdata = np.zeros(input_size, dtype = float)
    outdata[zs:ze,xs:xs+xl,ys:ys+yl] = np.absolute(normHighResRec1[:,:,:])
    maxValueOut = np.amax(outdata)
    maxValueLow = np.amax(np.absolute(lowRes1))
    maxValuePred = np.amax(normHighResRec1)
    print("max lowres, highres, predicted, out = ", maxValueLow, maxValueHR1, maxValuePred, maxValueOut)
    write2dicom(out_path1, out_path1, outdata, protocolName, nrExport)
    outdata[zs:ze,xs:xs+xl,ys:ys+yl] = np.absolute(normHighResRec2[:,:,:])
    write2dicom(out_path2, out_path2, outdata, protocolName, nrExport)
    outdata[zs:ze,xs:xs+xl,ys:ys+yl] = np.absolute(normHighResRec3[:,:,:])
    write2dicom(out_path3, out_path3, outdata, protocolName, nrExport)
    write2dicom(out_path_test1, out_path_test1, np.absolute(normHighResRecTest1), protocolName, nrExport)
    write2dicom(out_path_test2, out_path_test2, np.absolute(normHighResRecTest2), protocolName, nrExport)
    
# Plot training & validation loss values
# plot loss curves
plt.plot(x_value, train_loss_curve, 'r', label='training')
plt.plot(x_value, val_loss_curve, 'b', label='validation')
plt.title("Training loss (red) and validation loss (blue)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()

# rescale and calculate difference images
lowResref = lowResSelect1*maxValueHR1
highResref = highResSelect1*maxValueHR1
residual_train = highResref - lowResref
residual_predicted = normHighResRec1 - lowResref

# print model architecture
print("Model architecture:")
print(summary(myNetwork, input_size=(1,1,patch_size_x,patch_size_y)))

# clear GPU cache
del myNetwork
torch.cuda.empty_cache()

# plot some images
plt.figure(figsize=(16, 16))
plt.subplot(3,5,1).set_title('lowres')
plt.subplot(3,5,2).set_title('highres predicted')
plt.subplot(3,5,3).set_title('highres true')
plt.subplot(3,5,4).set_title('true residual')
plt.subplot(3,5,5).set_title('predicted residual')

for i in range(1,16,5):
    img_index = random.randint(0,len(lowResref)-1)
    maxValueT = np.amax(lowResref[img_index])
    print(i)
    
    # plot lowres image
    ax = plt.subplot(3,5,i)
    img1 = np.squeeze(lowResref[img_index])
    maxValue = np.amax(img1)
    img1 = img1*(maxValueT/maxValue)
    plt.imshow(img1, cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot estimated highres image
    ax = plt.subplot(3,5,i+1)
    img2 = np.squeeze(normHighResRec1[img_index])
    maxValue = np.amax(img2)
    img2 = img2*(maxValueT/maxValue)
    plt.imshow(img2, cmap=plt.cm.gray)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    # plot true highres image
    ax = plt.subplot(3,5,i+2)
    img3 = np.squeeze(highResref[img_index])
    maxValue = np.amax(img3)
    img3 = img3*(maxValueT/maxValue)
    plt.imshow(img3, cmap=plt.cm.gray)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    
    # plot true residual image
    ax = plt.subplot(3,5,i+3)
    img4 = np.squeeze(residual_train[img_index])
    maxValue = np.amax(img4)
    minValue = np.amin(img4)
    meanValue = np.mean(img4)
    img4 = img4 * 2.0
    print('true residual min max mean:', minValue, maxValue, meanValue)
    plt.imshow(img4, cmap=plt.cm.gray, vmin=-50.0, vmax=50.0)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    
    # plot predicted residual image
    ax = plt.subplot(3,5,i+4)
    img5 = np.squeeze(residual_predicted[img_index])
    maxValue = np.amax(img5)
    minValue = np.amin(img5)
    meanValue = np.mean(img5)
    img5 = img5 * 2.0
    print('estimated residual min max mean', minValue, maxValue, meanValue)
    plt.imshow(img5, cmap=plt.cm.gray, vmin=-50.0, vmax=50.0)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
plt.tight_layout()
