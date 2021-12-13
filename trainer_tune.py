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
from edgeloss import edge_loss2
import argparse
from dataset_tio import data_split, Normalize, calculate_overlap


class LitTrainer(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('LitModel')
        parser.add_argument('--learning_rate', type=float, default=1e-2)
        # parser.add_argument('--std', type=float, default=-.3548)
        return parent_parser

    def __init__(self,
                 netG,
                 config,
                 args,
                 # lr: float = 0.0002,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['netG'])
        self.learning_rate = config['learning_rate']

        self.netG = netG
        self.args = args
        self.criterion_edge = edge_loss2
        self.criterion_pixel = torch.nn.L1Loss()

    def make_grid(self, imgs_lr, imgs_hr, gen_hr):
        imgs_lr = torch.clamp((imgs_lr[:10]*self.args.std), 0, 1).squeeze()
        imgs_hr = torch.clamp((imgs_hr[:10]*self.args.std), 0, 1).squeeze()
        gen_hr = torch.clamp((gen_hr[:10]*self.args.std), 0, 1).squeeze()
        diff = (imgs_hr-gen_hr)*2 + .5

        img_grid = torch.cat([torch.stack([a, b, c, d]) for a, b, c, d in zip(imgs_lr, imgs_hr, gen_hr, diff)]).unsqueeze(1)

        tb_grid = torchvision.utils.make_grid(img_grid, nrow=4)
        return tb_grid

    def forward(self, inputs):
        return self.netG(inputs)

    def prepare_batch(self, batch):
        return batch['LR'][tio.DATA].squeeze(4), batch['HR'][tio.DATA].squeeze(4)

    def training_step(self, batch, batch_idx):
        imgs_lr, imgs_hr = self.prepare_batch(batch)

        # train generator
        self.gen_hr = self(imgs_lr)

        loss_edge = self.criterion_edge(self.gen_hr, imgs_hr)
        loss_pixel = self.criterion_pixel(self.gen_hr, imgs_hr)
        g_loss = 0.3 * loss_edge + 0.7 * loss_pixel

        self.log('Step loss/generator', {'train_loss_edge': loss_edge,
                                         'train_loss_pixel': loss_pixel,
                                         }, on_step=True, on_epoch=False, sync_dist=True)

        self.log('Epoch loss/generator', {'Train': g_loss,
                                          }, on_step=False, on_epoch=True, sync_dist=True)

        # if batch_idx % 100 == 0:
        #     grid = self.make_grid(imgs_lr, imgs_hr, self.gen_hr)
        #     self.logger.experiment.add_image('generated images/train', grid, batch_idx*(self.current_epoch+1), dataformats='CHW')

        return g_loss

    # def training_epoch_end(self, outputs):
    #     print(outputs)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs_lr, imgs_hr = self.prepare_batch(batch)
            gen_hr = self(imgs_lr)
            loss_edge = self.criterion_edge(gen_hr, imgs_hr)
            loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)
            g_loss = 0.3 * loss_edge + 0.7 * loss_pixel
            self.log('Epoch loss/generator', {'Val': g_loss}, on_step=False, on_epoch=True, sync_dist=True)

        # if batch_idx % 50 == 0:
        #     grid = self.make_grid(imgs_lr, imgs_hr, gen_hr)
        #     self.logger.experiment.add_image('generated images/val', grid, batch_idx*(self.current_epoch+1), dataformats='CHW')

        return g_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.stack(outputs))
        self.log("val_loss", avg_loss, sync_dist=True)

    def prepare_data(self):
        args = self.args
        train_subjects = data_split('training', patients_frac=args.patients_frac, root_dir=args.root_dir)
        val_subjects = data_split('validation', patients_frac=args.patients_frac, root_dir=args.root_dir)

        training_transform = tio.Compose([
            Normalize(std=args.std),
            # tio.RandomNoise(p=0.5),
            tio.RandomFlip(),
        ])

        self.training_set = tio.SubjectsDataset(
            train_subjects, transform=training_transform)

        self.val_set = tio.SubjectsDataset(
            val_subjects, transform=training_transform)

        overlap, nr_patches = calculate_overlap(train_subjects[0]['LR'],
                                                (args.patch_size, args.patch_size),
                                                (args.patch_overlap, args.patch_overlap)
                                                )
        self.samples_per_volume = nr_patches

        self.sampler = tio.data.GridSampler(patch_size=(args.patch_size, args.patch_size, 1),
                                            patch_overlap=overlap,
                                            # padding_mode=0,
                                            )

    def train_dataloader(self):
        training_queue = tio.Queue(
            subjects_dataset=self.training_set,
            max_length=self.samples_per_volume*100,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=self.args.num_workers,
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        training_loader = torch.utils.data.DataLoader(
            training_queue,
            batch_size=self.args.batch_size
        )
        return training_loader

    def val_dataloader(self):
        val_queue = tio.Queue(
            subjects_dataset=self.val_set,
            max_length=self.samples_per_volume*100,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=self.args.num_workers,
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_queue,
            batch_size=self.args.batch_size
        )
        return val_loader

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_g = torch.optim.SGD(self.netG.parameters(), lr=lr, momentum=0.9, nesterov=True)
        return {
            'optimizer': opt_g,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.99999),
            },
        }

