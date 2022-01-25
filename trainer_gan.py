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
from edgeloss import edge_loss1, edge_loss2, edge_loss3
import argparse
from dataset_tio import data_split, Normalize, calculate_overlap


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
            self.optimizer = torch.optim.Adam(netG.parameters(), lr=config['learning_rate'], betas=(config['b1'], config['b2']))

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
        self.criterion_GAN = GANLoss("wgangp")

        self.factor_content = config['content_alpha']
        self.factor_adversarial = config['adversarial_alpha']

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

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs_lr, imgs_hr = self.prepare_batch(batch)

        # train generator
        if optimizer_idx == 0:
            self.gen_hr = self(imgs_lr)

            # loss_edge = self.criterion_edge(self.gen_hr, imgs_hr)
            # loss_pixel = self.criterion_pixel(self.gen_hr, imgs_hr)

            # gen_features = self.netF(torch.repeat_interleave(self.gen_hr, 3, 1))
            # real_features = self.netF(torch.repeat_interleave(imgs_hr, 3, 1))  #.detach()
            # loss_content = self.criterion_content(gen_features, real_features)

            loss_adv = self.criterion_GAN(self.netD(self.gen_hr), True)

            # g_loss = 0.3 * loss_edge + 0.7 * loss_pixel + self.factor_content * loss_content + self.factor_adversarial * loss_adv

            if self.current_epoch < 5:
                g_loss = self.criterion_pixel(self.gen_hr, imgs_hr)
            else:
                g_loss = loss_adv

            # self.log('Step loss/generator', {'train_loss_edge': loss_edge,
            #                                  'train_loss_pixel': loss_pixel,
            #                                  }, on_step=True, on_epoch=False, sync_dist=True)

            self.log('Epoch loss/generator', {'Train': g_loss,
                                              }, on_step=False, on_epoch=True, sync_dist=True)

            if batch_idx % 20 == 0:
                grid = self.make_grid(imgs_lr, imgs_hr, self.gen_hr)
                self.logger.experiment.add_image('generated images/train', grid, batch_idx * (self.current_epoch + 1),
                                                 dataformats='CHW')

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

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs_lr, imgs_hr = self.prepare_batch(batch)
            gen_hr = self(imgs_lr)
            # loss_edge = self.criterion_edge(gen_hr, imgs_hr)
            # loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)
            loss_adv = self.criterion_GAN(self.netD(gen_hr), True)

            # gen_features = self.netF(torch.repeat_interleave(gen_hr, 3, 1))
            # real_features = self.netF(torch.repeat_interleave(imgs_hr, 3, 1))  # .detach()
            # loss_content = self.criterion_content(gen_features, real_features)

            # for real image
            pred_real = self.netD(imgs_hr)
            real_loss = self.criterion_GAN(pred_real, True)
            # for fake image
            pred_fake = self.netD(gen_hr.detach())
            fake_loss = self.criterion_GAN(pred_fake, False)

            d_loss = (real_loss + fake_loss) / 2

            # g_loss = 0.3 * loss_edge + 0.7 * loss_pixel + self.factor_content * loss_content + self.factor_adversarial * loss_adv
            g_loss = loss_adv
            self.log('Epoch loss/generator', {'Val': g_loss}, on_step=False, on_epoch=True, sync_dist=True)
            self.log('Epoch loss/discriminator', {"Val": d_loss}, on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx % 5 == 0:
            grid = self.make_grid(imgs_lr, imgs_hr, gen_hr)
            self.logger.experiment.add_image('generated images/val', grid, batch_idx*(self.current_epoch+1), dataformats='CHW')

        return g_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.stack(outputs))
        self.log('val_loss', avg_loss, sync_dist=True)

    def prepare_data(self):
        args = self.args
        data_path = os.path.join(args.root_dir, 'data')
        train_subjects = data_split('training', patients_frac=self.patients_frac, root_dir=data_path)
        val_subjects = data_split('validation', patients_frac=self.patients_frac, root_dir=data_path)

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
            max_length=self.samples_per_volume * 100,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=self.args.num_workers,
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        training_loader = torch.utils.data.DataLoader(
            training_queue,
            batch_size=self.batch_size
        )
        return training_loader

    def val_dataloader(self):
        val_queue = tio.Queue(
            subjects_dataset=self.val_set,
            max_length=self.samples_per_volume * 100,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=self.args.num_workers,
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_queue,
            batch_size=self.batch_size
        )
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
