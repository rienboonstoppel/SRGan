import os
import torchvision
from dataset_tio import *
from metrics import *
from torch.autograd import Variable
import sys
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from edgeloss import *
import torchio as tio
import pytorch_lightning as pl
import torch.nn.functional as F
from collections import OrderedDict
from generator import GeneratorRRDB
from discriminator import Discriminator
from lightning_losses import *
from edgeloss import edge_loss1
import argparse

class LitTrainer(pl.LightningModule):
    def __init__(self,
                 netG,
                 netD,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['netG', 'netD'])

        self.netG = netG
        self.netD = netD

        self.criterion_GAN = GANLoss('vanilla')
        self.criterion_edge = edge_loss1
        self.criterion_pixel = torch.nn.L1Loss()

    def forward(self, inputs):
        return self.netG(inputs)

    def prepare_batch(self, batch):
        return batch['LR'][tio.DATA].squeeze(4), batch['HR'][tio.DATA].squeeze(4)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs_lr, imgs_hr = self.prepare_batch(batch)

        # train generator
        if optimizer_idx == 0:
            self.gen_hr = self(imgs_lr)

            loss_adv = self.criterion_GAN(self.netD(self.gen_hr), True)
            loss_edge = self.criterion_edge(self.gen_hr, imgs_hr)
            loss_pixel = self.criterion_pixel(self.gen_hr, imgs_hr)
            g_loss = loss_adv + loss_edge + loss_pixel

            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})

            return output

        # train discriminator
        if optimizer_idx == 1:

            # for real image
            pred_real = self.netD(imgs_hr)
            real_loss = self.criterion_GAN(pred_real, True)
            # for fake image
            pred_fake = self.netD(self.gen_hr.detach())
            fake_loss = self.criterion_GAN(pred_fake, False)

            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_g = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

