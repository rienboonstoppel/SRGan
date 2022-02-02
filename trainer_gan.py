import os

import numpy as np
import pytorch_lightning as pl
import torch
# from metrics import *
# import sys
# from torchvision.utils import save_image
import torchio as tio
import torchvision

from dataset_tio import data_split, Normalize, calculate_overlap
from edgeloss import edge_loss1, edge_loss2, edge_loss3


class LitTrainer(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # parser.add_argument('--learning_rate', type=float, default=1e-2)
        # parser.add_argument('--std', type=float, default=-.3548)
        return parent_parser

    def __init__(self,
                 netG,
                 netF,
                 netD,
                 config,
                 args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['netG', 'netF', 'netD'])

        self.patients_frac = config['patients_frac']
        self.patch_overlap = config['patch_overlap']
        self.batch_size = config['batch_size']
        self.patch_size = config['patch_size']

        if config['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(netG.parameters(), lr=config['learning_rate'], momentum=0.9,
                                             nesterov=True)
        elif config['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(netG.parameters(), lr=config['learning_rate'],
                                              betas=(config['b1'], config['b2']))

        if config['edge_loss'] == 1:
            self.criterion_edge = edge_loss1
        elif config['edge_loss'] == 2:
            self.criterion_edge = edge_loss2
        elif config['edge_loss'] == 3:
            self.criterion_edge = edge_loss3

        self.netG = netG
        self.netF = netF.eval()
        self.netD = netD

        self.args = args

        self.criterion_pixel = torch.nn.L1Loss()
        self.criterion_content = torch.nn.L1Loss()
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss()

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    def make_grid(self, imgs_lr, imgs_hr, gen_hr):
        imgs_lr = torch.clamp((imgs_lr[:10] * self.args.std), 0, 1).squeeze()
        imgs_hr = torch.clamp((imgs_hr[:10] * self.args.std), 0, 1).squeeze()
        gen_hr = torch.clamp((gen_hr[:10] * self.args.std), 0, 1).squeeze()
        diff = (imgs_hr - gen_hr) * 2 + .5

        img_grid = torch.cat(
            [torch.stack([a, b, c, d]) for a, b, c, d in zip(imgs_lr, imgs_hr, gen_hr, diff)]).unsqueeze(1)

        tb_grid = torchvision.utils.make_grid(img_grid, nrow=4)
        return tb_grid

    def forward(self, inputs):
        return self.netG(inputs)

    def prepare_batch(self, batch):
        return batch['LR'][tio.DATA].squeeze(4), batch['HR'][tio.DATA].squeeze(4)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs_lr, imgs_hr = self.prepare_batch(batch)

        # # Configure model input
        # imgs_lr = torch.autograd.Variable(imgs_lr.type(self.Tensor))
        # imgs_hr = torch.autograd.Variable(imgs_hr.type(self.Tensor))

        # Adversarial ground truths
        self.valid = torch.autograd.Variable(self.Tensor(np.ones((imgs_lr.size(0), *self.netD.output_shape))),
                                             requires_grad=False)
        self.fake = torch.autograd.Variable(self.Tensor(np.zeros((imgs_lr.size(0), *self.netD.output_shape))),
                                            requires_grad=False)

        # ---------------------
        #  Train Generator
        # ---------------------
        if optimizer_idx == 0:
            train_batches_done = batch_idx + self.current_epoch * self.train_len
            gen_hr = self(imgs_lr)
            self.gen_hr = gen_hr
            loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)

            if train_batches_done < self.args.warmup_batches:
                # Warm-up (pixel loss only)
                g_loss = loss_pixel
                return g_loss

            loss_edge = self.criterion_edge(gen_hr, imgs_hr)

            gen_features = self.netF(torch.repeat_interleave(gen_hr, 3, 1))
            real_features = self.netF(torch.repeat_interleave(imgs_hr, 3, 1)).detach()
            loss_content = self.criterion_content(gen_features, real_features)

            # Extract validity predictions from discriminator
            pred_real = self.netD(imgs_hr).detach()
            pred_fake = self.netD(gen_hr)

            # Adversarial loss (relativistic average GAN)
            loss_adv = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), self.valid)

            g_loss = 0.3 * loss_edge + 0.7 * loss_pixel + 1 * loss_content + 1 * loss_adv

            self.log('Step loss/generator', {'train_loss_edge': loss_edge,
                                             'train_loss_pixel': loss_pixel,
                                             'train_loss_content': loss_content,
                                             'train_loss_adv': loss_adv,
                                             },
                     on_step=True,
                     on_epoch=False,
                     sync_dist=True,
                     prog_bar=True,
                     batch_size=self.batch_size)

            self.log('Epoch loss/generator', {'Train': g_loss,
                                              }, on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=self.batch_size)

            if train_batches_done % (self.args.log_every_n_steps * 5) == 0:
                grid = self.make_grid(imgs_lr, imgs_hr, self.gen_hr)
                self.logger.experiment.add_image('generated images/train', grid, train_batches_done,
                                                 dataformats='CHW')

            return g_loss

        # ---------------------
        #  Train Discriminator
        # ---------------------
        if optimizer_idx == 1:
            pred_real = self.netD(imgs_hr)
            pred_fake = self.netD(self.gen_hr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), self.valid)
            loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), self.fake)

            d_loss = (loss_real + loss_fake) / 2

            self.log('Epoch loss/discriminator', {"Train": d_loss}, on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=self.batch_size)

            return d_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            # ---------------------
            #  Validate Generator
            # ---------------------
            imgs_lr, imgs_hr = self.prepare_batch(batch)
            gen_hr = self(imgs_lr)

            loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)
            loss_edge = self.criterion_edge(gen_hr, imgs_hr)

            gen_features = self.netF(torch.repeat_interleave(gen_hr, 3, 1))
            real_features = self.netF(torch.repeat_interleave(imgs_hr, 3, 1))
            loss_content = self.criterion_content(gen_features, real_features)

            # Extract validity predictions from discriminator
            pred_real = self.netD(imgs_hr)
            pred_fake = self.netD(gen_hr)

            # Adversarial loss (relativistic average GAN)
            loss_adv = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), self.valid)

            g_loss = 0.3 * loss_edge + 0.7 * loss_pixel + 1 * loss_content + 1 * loss_adv

            # ---------------------
            #  Validate Discriminator
            # ---------------------
            pred_real = self.netD(imgs_hr)
            pred_fake = self.netD(gen_hr)

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), self.valid)
            loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), self.fake)

            d_loss = (loss_real + loss_fake) / 2

            self.log('Epoch loss/generator', {'Val': g_loss}, on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=self.batch_size)
            self.log('Epoch loss/discriminator', {"Val": d_loss}, on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=self.batch_size)

        val_batches_done = batch_idx + self.current_epoch * self.val_len
        if val_batches_done % self.args.log_every_n_steps == 0:
            grid = self.make_grid(imgs_lr, imgs_hr, gen_hr)
            self.logger.experiment.add_image('generated images/val', grid, val_batches_done,
                                             dataformats='CHW')

        return g_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.stack(outputs))
        self.log('val_loss', avg_loss, sync_dist=True)

    def setup(self, stage='fit'):
        args = self.args
        data_path = os.path.join(args.root_dir, 'data')
        train_subjects = data_split('training', patients_frac=self.patients_frac, root_dir=data_path)
        val_subjects = data_split('validation', patients_frac=self.patients_frac, root_dir=data_path)

        training_transform = tio.Compose([
            Normalize(std=args.std),
            # tio.RandomNoise(p=0.5),
            tio.RandomFlip(axes=(0, 1)),
        ])

        self.training_set = tio.SubjectsDataset(
            train_subjects, transform=training_transform)

        self.val_set = tio.SubjectsDataset(
            val_subjects, transform=training_transform)

        overlap, nr_patches = calculate_overlap(train_subjects[0]['LR'],
                                                (self.patch_size, self.patch_size),
                                                (self.patch_overlap, self.patch_overlap)
                                                )
        self.samples_per_volume = nr_patches

        self.sampler = tio.data.GridSampler(patch_size=(self.patch_size, self.patch_size, 1),
                                            patch_overlap=overlap,
                                            # padding_mode=0,
                                            )

    def train_dataloader(self):
        training_queue = tio.Queue(
            subjects_dataset=self.training_set,
            max_length=self.samples_per_volume * 10,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=self.args.num_workers,
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        training_loader = torch.utils.data.DataLoader(
            training_queue,
            batch_size=self.batch_size,
            num_workers=0,
        )
        self.train_len = len(training_loader)
        return training_loader

    def val_dataloader(self):
        val_queue = tio.Queue(
            subjects_dataset=self.val_set,
            max_length=self.samples_per_volume * 5,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=self.args.num_workers,
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_queue,
            batch_size=self.batch_size,
            num_workers=0,
        )
        self.val_len = len(val_loader)
        return val_loader

    def configure_optimizers(self):
        opt_g = self.optimizer
        opt_d = self.optimizer
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR

        return (
            {
                'optimizer': opt_g,
                'lr_scheduler':
                    {
                        'scheduler': lr_scheduler(self.optimizer, gamma=0.99999),
                    },
            },
            {
                'optimizer': opt_d,
                'lr_scheduler':
                    {
                        'scheduler': lr_scheduler(self.optimizer, gamma=0.99999),
                    },
            }
        )
