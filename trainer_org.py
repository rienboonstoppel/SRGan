import os
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import torchio as tio
import wandb
from torchvision.utils import make_grid
from edgeloss import edge_loss1, edge_loss2, edge_loss3
from dataset_tio import sim_data, calculate_overlap
from utils import imgs_cat, val_metrics
from transform import Normalize
warnings.filterwarnings('ignore', '.*The dataloader, .*')


class LitTrainer(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument('--std', type=float, default=0.3548)
        parser.add_argument('--middle_slices', type=int, default=100)
        parser.add_argument('--every_other', type=int, default=2)
        parser.add_argument('--sampler', type=str, default='label', choices=['grid', 'label'])
        return parent_parser

    def __init__(self,
                 netG, netF,
                 config,
                 args,
                 **kwargs
                 ):
        super().__init__()
        self.args = args
        # self.save_hyperparameters(ignore=['netG', 'netF', 'netD'])

        self.netG = netG
        self.netF = netF.eval()

        self.criterion_pixel = torch.nn.L1Loss()  # method to calculate pixel differences
        self.criterion_perceptual = torch.nn.L1Loss()  # method to calculate differences between vgg features
        self.criterion_edge = globals()['edge_loss' + str(config.edge_loss)]

        self.data_resolution = config.data_resolution
        self.patients_frac = config.patients_frac
        self.patch_overlap = config.patch_overlap
        self.batch_size = config.batch_size
        self.patch_size = config.patch_size
        self.alpha_edge = config.alpha_edge
        self.alpha_pixel = config.alpha_pixel
        self.alpha_perceptual = config.alpha_perceptual
        self.log_images_train = False
        self.log_images_val = True

        if config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(netG.parameters(),
                                             lr=config.learning_rate_G,
                                             momentum=0.9,
                                             nesterov=True)
        elif config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(netG.parameters(),
                                              lr=config.learning_rate_G,
                                              betas=(config.b1, config.b2))

    def forward(self, inputs):
        return self.netG(inputs)

    def prepare_batch(self, batch):
        return batch['LR'][tio.DATA].squeeze(4), batch['HR'][tio.DATA].squeeze(4)

    def training_step(self, batch, batch_idx):
        imgs_lr, imgs_hr = self.prepare_batch(batch)
        imgs_sr = self(imgs_lr)
        train_batches_done = batch_idx + self.current_epoch * self.train_len

        if train_batches_done < self.args.warmup_batches:
            loss_pixel = self.criterion_pixel(imgs_sr, imgs_hr)
            # Warm-up (pixel loss only)
            g_loss = loss_pixel
            return g_loss

        loss_edge = self.criterion_edge(imgs_sr, imgs_hr) if self.alpha_edge != 0 else 0
        loss_pixel = self.criterion_pixel(imgs_sr, imgs_hr) if self.alpha_pixel != 0 else 0

        if self.alpha_perceptual != 0:
            gen_features = self.netF(torch.repeat_interleave(imgs_sr, 3, 1))
            real_features = self.netF(torch.repeat_interleave(imgs_hr, 3, 1)).detach()
            loss_perceptual = self.criterion_perceptual(gen_features, real_features)
        else:
            loss_perceptual = 0

        g_loss = self.alpha_edge * loss_edge + \
                 self.alpha_pixel * loss_pixel + self.alpha_perceptual * loss_perceptual

        self.log('Generator train losses', {'edge': loss_edge,
                                            'pixel': loss_pixel,
                                            'perceptual': loss_perceptual,
                                            },
                 on_step=True, on_epoch=False, sync_dist=True, prog_bar=False, batch_size=self.batch_size)

        self.log('Generator train agg', {'loss': g_loss},
                 on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        if self.log_images_train:
            if train_batches_done % (self.args.log_every_n_steps * 5) == 0:
                grid = imgs_cat(imgs_lr * self.args.std, imgs_hr * self.args.std, imgs_sr * self.args.std)
                self.logger.log_image('images train', [wandb.Image(make_grid(torch.clamp(grid, 0, 1), nrow=10))])

        return g_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            with torch.no_grad():
                imgs_lr, imgs_hr = self.prepare_batch(batch)
                imgs_sr = self(imgs_lr)
                loss_edge = self.criterion_edge(imgs_sr, imgs_hr)
                loss_pixel = self.criterion_pixel(imgs_sr, imgs_hr)

                gen_features = self.netF(torch.repeat_interleave(imgs_sr, 3, 1))
                real_features = self.netF(torch.repeat_interleave(imgs_hr, 3, 1)).detach()
                loss_perceptual = self.criterion_perceptual(gen_features, real_features)

                g_loss = self.alpha_edge * loss_edge + self.alpha_pixel * loss_pixel + self.alpha_perceptual * loss_perceptual

                self.log('Generator val agg', {'loss': g_loss},
                         on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size,
                         add_dataloader_idx=False)

                if self.log_images_val:
                    val_batches_done = batch_idx + self.current_epoch * self.val_len
                    if val_batches_done % self.args.log_every_n_steps == 0:
                        grid = imgs_cat(imgs_lr * self.args.std, imgs_hr * self.args.std, imgs_sr * self.args.std)
                        self.logger.log_image('images validation',
                                              [wandb.Image(make_grid(torch.clamp(grid, 0, 1), nrow=10))])

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
        self.log('SSIM_mean', metrics['SSIM']['mean'], sync_dist=True)

        self.log('NCC', {'Mean': metrics['NCC']['mean'],
                         'Q1': metrics['NCC']['quartiles'][0],
                         'Median': metrics['NCC']['quartiles'][1],
                         'Q3': metrics['NCC']['quartiles'][2],
                         },
                 on_epoch=True, sync_dist=True, prog_bar=False, batch_size=self.batch_size)

        self.log('NRMSE', {'Mean': metrics['NRMSE']['mean'],
                           'Q1': metrics['NRMSE']['quartiles'][0],
                           'Median': metrics['NRMSE']['quartiles'][1],
                           'Q3': metrics['NRMSE']['quartiles'][2],
                           },
                 on_epoch=True, sync_dist=True, prog_bar=False, batch_size=self.batch_size)

        middle = int(SR_aggs[0].shape[3] / 2)
        grid = torch.stack([SR_aggs[i][:, :, :, middle].squeeze() for i in range(len(SR_aggs))], dim=0).unsqueeze(1)
        self.logger.log_image('aggregated validation', [wandb.Image(make_grid(torch.clamp(grid, 0, 1), nrow=5))])

    def setup(self, stage='fit'):
        args = self.args
        data_path = os.path.join(args.root_dir, 'data')
        train_subjects = sim_data(dataset='training',
                                  patients_frac=self.patients_frac,
                                  root_dir=data_path,
                                  data_resolution=self.data_resolution,
                                  middle_slices=args.middle_slices,
                                  every_other=args.every_other)
        val_subjects = sim_data(dataset='validation',
                                patients_frac=self.patients_frac,
                                root_dir=data_path,
                                data_resolution=self.data_resolution,
                                middle_slices=args.middle_slices,
                                every_other=args.every_other)
        test_subjects = sim_data(dataset='test',
                                 patients_frac=self.patients_frac,
                                 root_dir=data_path,
                                 data_resolution=self.data_resolution,
                                 middle_slices=args.middle_slices,
                                 every_other=args.every_other)

        # train_subjects = mixed_data(dataset='training', combined_num_patients=self.num_patients, num_real=self.num_real, root_dir=data_path)
        # val_subjects = mixed_data(dataset='validation', combined_num_patients=self.num_patients, num_real=self.num_real, root_dir=data_path)
        # test_subjects = mixed_data(dataset='test', combined_num_patients=self.num_patients, num_real=self.num_real,
        #                            numslices=45, root_dir=data_path)

        self.num_test_subjects = len(test_subjects)

        self.post_proc_info = []
        for subject in test_subjects:
            mask = subject['MSK'].data

            bg_idx = np.where(mask == 0)
            brain_idx = np.where(mask.squeeze(0) != 0)
            crop_coords = ([brain_idx[i].min() for i in range(len(brain_idx))],
                           [brain_idx[i].max() for i in range(len(brain_idx))])
            self.post_proc_info.append((bg_idx, crop_coords))

        training_transform = tio.Compose([
            Normalize(std=args.std),
            # tio.RandomNoise(p=0.5),
            tio.RandomFlip(axes=(0, 1), flip_probability=0.5),
            tio.RandomFlip(axes=(0, 1), flip_probability=0.75),
        ])

        test_transform = tio.Compose([Normalize(std=args.std), ])

        self.training_set = tio.SubjectsDataset(
            train_subjects, transform=training_transform)

        self.val_set = tio.SubjectsDataset(
            val_subjects, transform=training_transform)

        self.test_set = tio.SubjectsDataset(
            test_subjects, transform=test_transform)

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
        for i in range(self.num_test_subjects):
            grid_sampler = tio.inference.GridSampler(
                subject=self.test_set[i],
                patch_size=(self.patch_size, self.patch_size, 1),
                patch_overlap=self.overlap,
                padding_mode=0,
            )
            test_loader = torch.utils.data.DataLoader(
                grid_sampler, batch_size=self.batch_size)
            loaders.append(test_loader)

        self.aggregator_HR = tio.inference.GridAggregator(grid_sampler)
        self.aggregator_SR = tio.inference.GridAggregator(grid_sampler)

        self.test_len = len(self.test_set)

        return [val_loader, *loaders]

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99999)}
        }
