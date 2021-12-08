import os
import torch
import torchvision
# from metrics import *
# import sys
# from torchvision.utils import save_image
from edgeloss import edge_loss1
import torchio as tio
import pytorch_lightning as pl
from lightning_losses import GANLoss
from edgeloss import edge_loss1
import argparse

class LitTrainer(pl.LightningModule):
    def __init__(self,
                 netG,
                 netD,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 std: float = 0.3548,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["netG", "netD"])

        self.netG = netG
        self.netD = netD

        self.criterion_GAN = GANLoss("vanilla")
        self.criterion_edge = edge_loss1
        self.criterion_pixel = torch.nn.L1Loss()


    def make_grid(self, imgs_lr, imgs_hr, gen_hr):
        imgs_lr = torch.clamp((imgs_lr[:10]*self.hparams.std), 0, 1).squeeze()
        imgs_hr = torch.clamp((imgs_hr[:10]*self.hparams.std), 0, 1).squeeze()
        gen_hr = torch.clamp((gen_hr[:10]*self.hparams.std), 0, 1).squeeze()
        diff = imgs_hr-gen_hr + .5

        img_grid = torch.cat([torch.stack([a, b, c, d]) for a, b, c, d in zip(imgs_lr, imgs_hr, gen_hr, diff)]).unsqueeze(1)

        tb_grid = torchvision.utils.make_grid(img_grid, nrow=4)
        return tb_grid

    def forward(self, inputs):
        return self.netG(inputs)

    def prepare_batch(self, batch):
        return batch["LR"][tio.DATA].squeeze(4), batch["HR"][tio.DATA].squeeze(4)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs_lr, imgs_hr = self.prepare_batch(batch)

        # train generator
        if optimizer_idx == 0:
            self.gen_hr = self(imgs_lr)

            loss_adv = self.criterion_GAN(self.netD(self.gen_hr), True)
            loss_edge = self.criterion_edge(self.gen_hr, imgs_hr)
            loss_pixel = self.criterion_pixel(self.gen_hr, imgs_hr)
            g_loss = .01 * loss_adv + loss_edge + loss_pixel + loss_edge*loss_pixel

            self.log('Step loss/generator', {'train_loss_adv': loss_adv,
                                             'train_loss_edge': loss_edge,
                                             'train_loss_pixel': loss_pixel,
                                             }, on_step=True, on_epoch=False, sync_dist=True)

            self.log('Epoch loss/generator', {"Train": g_loss}, on_step=False, on_epoch=True, sync_dist=True)

            if batch_idx % 10 == 0:
                grid = self.make_grid(imgs_lr, imgs_hr, self.gen_hr)
                self.logger.experiment.add_image('generated images/train', grid, batch_idx*(self.current_epoch+1), dataformats='CHW')

            return g_loss

        # train discriminator
        if optimizer_idx == 1:

            # for real image
            pred_real = self.netD(imgs_hr)
            real_loss = self.criterion_GAN(pred_real, True)
            # for fake image
            pred_fake = self.netD(self.gen_hr.detach())
            fake_loss = self.criterion_GAN(pred_fake, False)

            d_loss = (real_loss + fake_loss) / 2

            self.log('Epoch loss/discriminator', {"Train": d_loss}, on_step=False, on_epoch=True, sync_dist=True)

            return d_loss

    # def training_epoch_end(self, outputs):
    #     print(outputs)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs_lr, imgs_hr = self.prepare_batch(batch)
            gen_hr = self(imgs_lr)
            loss_adv = self.criterion_GAN(self.netD(gen_hr), True)
            loss_edge = self.criterion_edge(gen_hr, imgs_hr)
            loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)
            g_loss = .01 * loss_adv + loss_edge + loss_pixel + loss_edge*loss_pixel

            # for real image
            pred_real = self.netD(imgs_hr)
            real_loss = self.criterion_GAN(pred_real, True)
            # for fake image
            pred_fake = self.netD(gen_hr.detach())
            fake_loss = self.criterion_GAN(pred_fake, False)

            d_loss = (real_loss + fake_loss) / 2

            self.log('Epoch loss/generator', {"Val": g_loss}, on_step=False, on_epoch=True, sync_dist=True)
            self.log('Epoch loss/discriminator', {"Val": d_loss}, on_step=False, on_epoch=True, sync_dist=True)

            if batch_idx % 10 == 0:
                grid = self.make_grid(imgs_lr, imgs_hr, gen_hr)
                self.logger.experiment.add_image('generated images/val', grid, batch_idx*(self.current_epoch+1), dataformats='CHW')

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_g = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

