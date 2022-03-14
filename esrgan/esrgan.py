"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instruction on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the script using command 'python3 esrgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import ImageDataset
from dataset_tio import data_split, Normalize, calculate_overlap
import torchvision.transforms as transforms
import torchio as tio
from PIL import Image
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
parser.add_argument("--patch_size", type=int, required=True)
parser.add_argument("--name", type=str, required=True)
opt = parser.parse_args()
print(opt)

os.makedirs(opt.name+"/images/training", exist_ok=True)
os.makedirs(opt.name+"/saved_models", exist_ok=True)
os.makedirs(opt.name+"/log", exist_ok=True)

writer = SummaryWriter(opt.name+"/log")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks, num_upsample=0).to(device)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
feature_extractor = FeatureExtractor().to(device)

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load(opt.name+"/saved_models/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load(opt.name+"/saved_models/discriminator_%d.pth" % opt.epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

root_dir = '/mnt/beta/djboonstoppel/Code'
std = 0.3548
patch_size = opt.patch_size
patch_overlap = .5

data_path = os.path.join(root_dir, 'data')
train_subjects = data_split('training', patients_frac=1, root_dir=data_path)

training_transform = tio.Compose([
    Normalize(std=std),
    # tio.RandomNoise(p=0.5),
    tio.RandomFlip(axes=(0, 1)),
])

training_set = tio.SubjectsDataset(
    train_subjects, transform=training_transform)

overlap, nr_patches = calculate_overlap(train_subjects[0]['LR'],
                                        (patch_size, patch_size),
                                        (patch_overlap, patch_overlap)
                                        )

sampler = tio.data.GridSampler(patch_size=(patch_size, patch_size, 1),
                                    patch_overlap=overlap,
                                    # padding_mode=0,
                                    )

training_queue = tio.Queue(
    subjects_dataset=training_set,
    max_length=nr_patches * 10,
    samples_per_volume=nr_patches,
    sampler=sampler,
    num_workers=8,
    shuffle_subjects=True,
    shuffle_patches=True,
)
dataloader = torch.utils.data.DataLoader(
    training_queue,
    batch_size=opt.batch_size,
    num_workers=0,
)

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        batches_done = epoch * len(dataloader) + i

        # Configure model input
        imgs_lr = imgs['LR'][tio.DATA].squeeze(4)
        imgs_hr = imgs['HR'][tio.DATA].squeeze(4)

        imgs_lr = Variable(imgs_lr.type(Tensor))
        imgs_hr = Variable(imgs_hr.type(Tensor))

        # imgs_lr = transforms.functional.resize(imgs_lr, 16,
        #                                        interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        if batches_done < opt.warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel.item()), end="\r"
            )
            writer.add_scalars(
                'Generator batch loss', {'Train': loss_pixel,
                                         'Train_pixel': loss_pixel,
                                         }, batches_done)
            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        gen_features = feature_extractor(torch.repeat_interleave(gen_hr, 3, 1))
        real_features = feature_extractor(torch.repeat_interleave(imgs_hr, 3, 1)).detach()
        loss_content = criterion_content(gen_features, real_features)

        # Total generator loss
        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        writer.add_scalars(
            'Generator batch loss', {'Train': loss_G,
                                     'Train_pixel': loss_pixel,
                                     'Train_adversarial': loss_GAN,
                                     'Train_content': loss_content,
                                     }, batches_done)

        writer.add_scalars(
            'Discriminator batch loss', {'Train': loss_D,
                                         }, batches_done)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_content.item(),
                loss_GAN.item(),
                loss_pixel.item(),
            ),
            end="\r"
        )

        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and ESRGAN outputs
            # imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            diff = (imgs_hr - gen_hr) * 2 + .5
            img_grid = torch.cat((imgs_lr*std, imgs_hr*std, gen_hr*std, diff), -1)[:10]

            save_image(img_grid, opt.name+"/images/training/%d.png" % batches_done, nrow=1, normalize=False)
            writer.add_image("Training", make_grid(torch.clamp(img_grid, 0, 1), nrow=1), batches_done)

        if batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), opt.name+"/saved_models/generator_%d.pth" % epoch)
            torch.save(discriminator.state_dict(), opt.name+"/saved_models/discriminator_%d.pth" % epoch)
