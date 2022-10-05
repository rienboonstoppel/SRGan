import os
import warnings
import numpy as np
import pytorch_lightning as pl
import torch
import torchio as tio
import wandb
from torchvision.utils import make_grid
from edgeloss import edge_loss1, edge_loss2, edge_loss3
from dataset_tio import sim_data, calculate_overlap, HCP_data, data
from transform import Normalize, RandomGamma, RandomIntensity, RandomBiasField, RandomBlur

from lightning_losses import GANLoss, GradientPenalty
from utils import val_metrics, imgs_cat

warnings.filterwarnings('ignore', '.*The dataloader, .*')


class LitTrainer(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument('--std', type=float, default=0.3548)
        parser.add_argument('--middle_slices', type=int, default=100)
        parser.add_argument('--every_other', type=int, default=1)
        parser.add_argument('--sampler', type=str, default='label', choices=['grid', 'label'])
        return parent_parser

    def __init__(self,
                 netG, netF, netD,
                 config,
                 args,
                 **kwargs
                 ):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=['netG', 'netF', 'netD'])

        self.netG = netG
        self.netF = netF.eval()
        self.netD = netD

        self.criterion_pixel = torch.nn.L1Loss()  # method to calculate pixel differences
        self.criterion_perceptual = torch.nn.L1Loss()  # method to calculate differences between vgg features
        self.criterion_GAN = GANLoss(gan_mode=config.gan_mode)  # method to calculate adversarial loss
        self.gradient_penalty = GradientPenalty(critic=self.netD, fake_label=1.0)
        self.criterion_edge = globals()['edge_loss' + str(config.edge_loss)]
        self.alpha_edge = config.alpha_edge
        self.alpha_pixel = config.alpha_pixel
        self.alpha_perceptual = config.alpha_perceptual
        self.alpha_adv = config.alpha_adversarial

        self.netD_freq = config.netD_freq

        self.data_resolution = config.data_resolution
        self.nr_sim_train = config.nr_sim_train
        self.nr_hcp_train = config.nr_hcp_train
        self.patch_overlap = config.patch_overlap
        self.batch_size = config.batch_size
        self.patch_size = config.patch_size
        self.ragan = config.ragan
        self.log_images_train = False
        self.log_images_val = False

        if config.optimizer == 'sgd':
            self.optimizer_G = torch.optim.SGD(netG.parameters(),
                                               lr=config.learning_rate_G,
                                               momentum=0.9,
                                               nesterov=True)
            self.optimizer_D = torch.optim.SGD(netD.parameters(),
                                               lr=config.learning_rate_D,
                                               momentum=0.9,
                                               nesterov=True)

        elif config.optimizer == 'adam':
            self.optimizer_G = torch.optim.Adam(netG.parameters(),
                                                lr=config.learning_rate_G,
                                                betas=(config.b1, config.b2))
            self.optimizer_D = torch.optim.Adam(netD.parameters(),
                                                lr=config.learning_rate_D,
                                                betas=(config.b1, config.b2))

    def forward(self, inputs):
        return self.netG(inputs)

    def prepare_batch(self, batch):
        return batch['LR'][tio.DATA].squeeze(4), batch['HR'][tio.DATA].squeeze(4)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs_lr, imgs_hr = self.prepare_batch(batch)
        imgs_sr = self(imgs_lr)
        train_batches_done = batch_idx + self.current_epoch * self.train_len

        if self.log_images_train:
            if train_batches_done % (self.args.log_every_n_steps * 5) == 0:
                grid = imgs_cat(imgs_lr * self.args.std, imgs_hr * self.args.std, imgs_sr * self.args.std)
                self.logger.log_image('images train', [wandb.Image(make_grid(torch.clamp(grid, 0, 1.5), nrow=10))])

        # ---------------------
        #  Train Generator
        # ---------------------
        if optimizer_idx == 0:

            if train_batches_done < self.args.warmup_batches:
                loss_pixel = self.criterion_pixel(imgs_sr, imgs_hr)
                # Warm-up (pixel loss only)
                g_loss = loss_pixel
                return g_loss

            loss_pixel = self.criterion_pixel(imgs_sr, imgs_hr) if self.alpha_pixel != 0 else 0
            loss_edge = self.criterion_edge(imgs_sr, imgs_hr) if self.alpha_edge != 0 else 0

            if self.alpha_perceptual != 0:
                gen_features = self.netF(torch.repeat_interleave(imgs_sr, 3, 1))
                real_features = self.netF(torch.repeat_interleave(imgs_hr, 3, 1)).detach()
                loss_perceptual = self.criterion_perceptual(gen_features, real_features)
            else:
                loss_perceptual = 0

            # Extract validity predictions from discriminator
            pred_real = self.netD(imgs_hr).detach()
            pred_fake = self.netD(imgs_sr)

            if self.ragan:
                pred_fake -= pred_real.mean(0, keepdim=True)

            # Adversarial loss
            loss_adv = self.criterion_GAN(pred_fake, True)
            # Calculate gradient penalty
            # gradient_penalty = self.gradient_penalty(imgs_hr, imgs_sr)

            g_loss = self.alpha_edge * loss_edge + \
                     self.alpha_pixel * loss_pixel + \
                     self.alpha_adv * loss_adv + \
                     self.alpha_perceptual * loss_perceptual

            self.log('Generator train losses', {'edge': loss_edge,
                                                'pixel': loss_pixel,
                                                'perceptual': loss_perceptual,
                                                'adversarial': loss_adv,
                                                # 'gradient_penalty': gradient_penalty,
                                                },
                     on_step=True, on_epoch=False, sync_dist=True, prog_bar=False, batch_size=self.batch_size)

            self.log('Generator train agg', {'loss': g_loss},
                     on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

            return g_loss

        # ---------------------
        #  Train Discriminator
        # ---------------------
        if optimizer_idx == 1:
            # Extract validity predictions from discriminator
            pred_real = self.netD(imgs_hr)
            pred_fake = self.netD(imgs_sr.detach())

            if self.ragan:
                pred_real, pred_fake = pred_real - pred_fake.mean(0, keepdim=True), \
                                       pred_fake - pred_real.mean(0, keepdim=True)

            # Adversarial loss
            loss_real = self.criterion_GAN(pred_real, True)
            loss_fake = self.criterion_GAN(pred_fake, False)

            d_loss = (loss_real + loss_fake) / 2

            self.log('Discriminator train', {'loss': d_loss},
                     on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

            return d_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            # ---------------------
            #  Validate Generator
            # ---------------------
            imgs_lr, imgs_hr = self.prepare_batch(batch)
            imgs_sr = self(imgs_lr)
            loss_pixel = self.criterion_pixel(imgs_sr, imgs_hr) if self.alpha_pixel != 0 else 0
            loss_edge = self.criterion_edge(imgs_sr, imgs_hr) if self.alpha_edge != 0 else 0

            if self.alpha_perceptual != 0:
                gen_features = self.netF(imgs_sr.repeat(1, 3, 1, 1))
                real_features = self.netF(imgs_hr.repeat(1, 3, 1, 1))
                loss_perceptual = self.criterion_perceptual(gen_features, real_features)
            else:
                loss_perceptual = 0

            # Extract validity predictions from discriminator
            pred_real = self.netD(imgs_hr)
            pred_fake = self.netD(imgs_sr)

            # Relativistic average GAN
            if self.ragan:
                pred_real, pred_fake = pred_real - pred_fake.mean(0, keepdim=True), \
                                       pred_fake - pred_real.mean(0, keepdim=True)

            # Adversarial loss
            loss_adv = self.criterion_GAN(pred_fake, True)  # Gradient Penalty cannot be calculated during validation

            g_loss = self.alpha_edge * loss_edge + \
                     self.alpha_pixel * loss_pixel + \
                     self.alpha_adv * loss_adv + \
                     self.alpha_perceptual * loss_perceptual

            # ---------------------
            #  Validate Discriminator
            # ---------------------

            # Adversarial loss for real and fake images
            loss_real = self.criterion_GAN(pred_real, True)
            loss_fake = self.criterion_GAN(pred_fake, False)

            d_loss = (loss_real + loss_fake) / 2

            self.log('Generator val agg', {'loss': g_loss},
                     on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size, add_dataloader_idx=False)
            self.log('Discriminator val', {'loss': d_loss},
                     on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size, add_dataloader_idx=False)

            if self.log_images_val:
                val_batches_done = batch_idx + self.current_epoch * self.val_len
                if val_batches_done % self.args.log_every_n_steps == 0:
                    grid = imgs_cat(imgs_lr * self.args.std, imgs_hr * self.args.std, imgs_sr * self.args.std)
                    self.logger.log_image('images validation',
                                          [wandb.Image(make_grid(torch.clamp(grid, 0, 1.5), nrow=10))])

            return g_loss

        if dataloader_idx > 0:
            imgs_lr, imgs_hr = self.prepare_batch(batch)
            locations = batch[tio.LOCATION]
            imgs_sr = self(imgs_lr)
            return imgs_hr, imgs_sr, locations

    def validation_epoch_end(self, outputs):
        val_loss = outputs[0]
        avg_loss = torch.mean(torch.stack(val_loss))
        self.log('val_loss', avg_loss, sync_dist=True)

        output_data = outputs[1:]

        SR_aggs, metrics = val_metrics(output_data, self.aggregator_HR, self.aggregator_SR, self.args.std,
                                       self.post_proc_info)

        self.log('SSIM', {'Mean': metrics['SSIM']['mean'],
                          'Q1': metrics['SSIM']['quartiles'][0],
                          'Median': metrics['SSIM']['quartiles'][1],
                          'Q3': metrics['SSIM']['quartiles'][2],
                          },
                 on_epoch=True, sync_dist=True, prog_bar=False, batch_size=self.batch_size)
        self.log('SSIM_mean', metrics['SSIM']['mean'], sync_dist=True, prog_bar=True)

        self.log('NCC', {'Mean': metrics['NCC']['mean'],
                         'Q1': metrics['NCC']['quartiles'][0],
                         'Median': metrics['NCC']['quartiles'][1],
                         'Q3': metrics['NCC']['quartiles'][2],
                         },
                 on_epoch=True, sync_dist=True, prog_bar=False, batch_size=self.batch_size)
        self.log('NCC_mean', metrics['NCC']['mean'], sync_dist=True, prog_bar=True)

        self.log('NRMSE', {'Mean': metrics['NRMSE']['mean'],
                           'Q1': metrics['NRMSE']['quartiles'][0],
                           'Median': metrics['NRMSE']['quartiles'][1],
                           'Q3': metrics['NRMSE']['quartiles'][2],
                           },
                 on_epoch=True, sync_dist=True, prog_bar=False, batch_size=self.batch_size)

        middle = int(SR_aggs[0].shape[3] / 2)
        grid = torch.stack([SR_aggs[i][:, :, :, middle].squeeze() for i in range(len(SR_aggs))], dim=0).unsqueeze(1)
        self.logger.log_image('aggregated validation', [wandb.Image(make_grid(torch.clamp(grid, 0, 1.5), nrow=5))])

    def setup(self, stage='fit'):
        args = self.args
        data_path = os.path.join(args.root_dir, 'data')

        train_subjects = data(dataset='training',
                              root_dir=data_path,
                              nr_sim=self.nr_sim_train,
                              nr_hcp=self.nr_hcp_train,
                              middle_slices=args.middle_slices,
                              every_other=args.every_other)
        # val_subjects = data(dataset='validation',
        #                     root_dir=data_path,
        #                     middle_slices=args.middle_slices,
        #                     every_other=args.every_other)

        val_subjects, _ = sim_data(dataset='validation',
                                   middle_slices=args.middle_slices,
                                   root_dir=data_path,
                                   every_other=args.every_other)

        self.num_val_subjects = len(val_subjects)

        self.post_proc_info = []
        for subject in val_subjects:
            mask = subject['MSK'].data

            bg_idx = np.where(mask == 0)
            brain_idx = np.where(mask.squeeze(0) != 0)
            crop_coords = ([brain_idx[i].min() for i in range(len(brain_idx))],
                           [brain_idx[i].max() for i in range(len(brain_idx))])
            self.post_proc_info.append((bg_idx, crop_coords))

        training_transform = tio.Compose([
            RandomBiasField(coefficients=0.2, p=0.5),
            RandomGamma(p=0.5),
            RandomIntensity(intensity_diff=(-0.3, 0.2), p=0.5),
            # RandomBlur(std=(0,1), p=0.75),
            Normalize(std=args.std, p=1),
            # tio.RandomNoise(p=0.5),
            tio.RandomFlip(axes=(0, 1), flip_probability=0.5),
            tio.RandomFlip(axes=(0, 1), flip_probability=0.75),
        ])

        val_agg_transform = tio.Compose([
            RandomBiasField(coefficients=0.3),
            RandomGamma(),
            RandomIntensity(),
            # RandomBlur(std=1, p=0.75),
            Normalize(std=args.std),
            # tio.RandomNoise(p=0.5),
            # tio.RandomFlip(axes=(0, 1), flip_probability=0.5),
            # tio.RandomFlip(axes=(0, 1), flip_probability=0.75),
        ])

        # test_transform = tio.Compose([Normalize(std=args.std), ])

        self.training_set = tio.SubjectsDataset(
            train_subjects, transform=training_transform)

        self.val_set = tio.SubjectsDataset(
            val_subjects, transform=training_transform)

        self.val_set_agg = tio.SubjectsDataset(
            val_subjects, transform=val_agg_transform)

        self.overlap, self.samples_per_volume = calculate_overlap(train_subjects[0]['LR'],
                                                                  (self.patch_size, self.patch_size),
                                                                  (self.patch_overlap, self.patch_overlap)
                                                                  )
        if args.sampler == 'grid':
            self.sampler = tio.data.GridSampler(patch_size=(self.patch_size, self.patch_size, 1),
                                                patch_overlap=self.overlap,
                                                padding_mode=None,
                                                )
        elif args.sampler == 'label':
            probabilities = {0: 0, 1: 1}
            self.sampler = tio.data.LabelSampler(
                patch_size=(self.patch_size, self.patch_size, 1),
                label_name='MSK',
                label_probabilities=probabilities,
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

        loaders = []
        for i in range(10): #range(self.num_val_subjects):
            grid_sampler = tio.inference.GridSampler(
                subject=self.val_set_agg[i],
                patch_size=(self.patch_size, self.patch_size, 1),
                patch_overlap=self.overlap,
                padding_mode=0,
            )
            val_agg_loader = torch.utils.data.DataLoader(
                grid_sampler, batch_size=self.batch_size)
            loaders.append(val_agg_loader)

        self.aggregator_HR = tio.inference.GridAggregator(grid_sampler)
        self.aggregator_SR = tio.inference.GridAggregator(grid_sampler)

        self.val_agg_len = len(self.val_set_agg)

        return [val_loader, *loaders]

    def configure_optimizers(self):
        opt_g = self.optimizer_G
        opt_d = self.optimizer_D
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR

        return (
            {'optimizer': opt_g,
             'lr_scheduler': {'scheduler': lr_scheduler(self.optimizer_G, gamma=0.99999)},
             'frequency': 1},
            {'optimizer': opt_d,
             'lr_scheduler': {'scheduler': lr_scheduler(self.optimizer_D, gamma=0.99999)},
             'frequency': self.netD_freq})
