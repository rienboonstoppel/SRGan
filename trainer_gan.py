import os
import numpy as np
import pytorch_lightning as pl
import torch
import torchio as tio
import torchvision
from lightning_losses import GANLoss, GradientPenalty
from dataset_tio import data_split, Normalize, calculate_overlap
from edgeloss import edge_loss1, edge_loss2, edge_loss3
from torchvision.utils import save_image, make_grid
import warnings

warnings.filterwarnings('ignore', '.*The dataloader, .*')


class LitTrainer(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # parser.add_argument('--learning_rate', type=float, default=1e-2)
        # parser.add_argument('--std', type=float, default=-.3548)
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
        self.criterion_content = torch.nn.L1Loss()  # method to calculate differences between vgg features
        self.criterion_GAN = GANLoss(gan_mode='vanilla')  # method to calculate adversarial loss
        self.gradient_penalty = GradientPenalty(critic=self.netD, fake_label=1.0)
        self.criterion_edge = globals()['edge_loss' + str(config['edge_loss'])]
        self.alpha_adv = config['alpha_adversarial']
        self.netD_freq = config['netD_freq']

        self.patients_frac = config['patients_frac']
        self.patch_overlap = config['patch_overlap']
        self.batch_size = config['batch_size']
        self.patch_size = config['patch_size']
        self.ragan = config['ragan']
        self.log_images_train = False
        self.log_images_val = True

        if config['optimizer'] == 'sgd':
            self.optimizer_G = torch.optim.SGD(netG.parameters(),
                                             lr=config['learning_rate'],
                                             momentum=0.9,
                                             nesterov=True)
        elif config['optimizer'] == 'adam':
            self.optimizer_G = torch.optim.Adam(netG.parameters(),
                                              lr=config['learning_rate'],
                                              betas=(config['b1'], config['b2']))
        if config['optimizer'] == 'sgd':
            self.optimizer_D = torch.optim.SGD(netD.parameters(),
                                             lr=config['learning_rate'],
                                             momentum=0.9,
                                             nesterov=True)
        elif config['optimizer'] == 'adam':
            self.optimizer_D = torch.optim.Adam(netD.parameters(),
                                              lr=config['learning_rate'],
                                              betas=(config['b1'], config['b2']))

    def imgs_cat(self, imgs_lr, imgs_hr, imgs_sr):
        imgs_lr = (imgs_lr[:10] * self.args.std).squeeze()
        imgs_hr = (imgs_hr[:10] * self.args.std).squeeze()
        imgs_sr = (imgs_sr[:10] * self.args.std).squeeze()
        diff = (imgs_hr - imgs_sr) * 2 + .5
        img_grid = torch.cat(
            [torch.stack([a, b, c, d]) for a, b, c, d in zip(imgs_lr, imgs_hr, imgs_sr, diff)]).unsqueeze(1)
        return img_grid

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
                grid = self.imgs_cat(imgs_lr, imgs_hr, imgs_sr)
                self.logger.experiment.add_image('generated images/train',
                                                 make_grid(torch.clamp(grid, 0, 1), nrow=4),
                                                 train_batches_done,
                                                 dataformats='CHW')
                save_dir = os.path.join(self.logger.log_dir, 'images', 'training')
                os.makedirs(save_dir, exist_ok=True)
                save_image(grid, save_dir + "/%04d.png" % train_batches_done, nrow=4, normalize=False)
                inference_path = os.path.join(save_dir, 'inference')
                os.makedirs(inference_path + "/%04d" % train_batches_done, exist_ok=True)
                for i in range(1):
                    img_sr = imgs_sr[i, 0, :, :] * self.args.std
                    save_image(img_sr, os.path.join(inference_path, "%04d" % train_batches_done, '%04d.png' % i))
                    np.savetxt(os.path.join(inference_path, "%04d" % train_batches_done, '%04d.csv' % i),
                               img_sr.detach().cpu().numpy(), delimiter=',')

        # ---------------------
        #  Train Generator
        # ---------------------
        if optimizer_idx == 0:
            loss_pixel = self.criterion_pixel(imgs_sr, imgs_hr)

            if train_batches_done < self.args.warmup_batches:
                # Warm-up (pixel loss only)
                g_loss = loss_pixel
                return g_loss

            loss_edge = self.criterion_edge(imgs_sr, imgs_hr)

            # gen_features = self.netF(torch.repeat_interleave(imgs_sr, 3, 1))
            # real_features = self.netF(torch.repeat_interleave(imgs_hr, 3, 1)).detach()
            # loss_content = self.criterion_content(gen_features, real_features)

            # Extract validity predictions from discriminator
            pred_real = self.netD(imgs_hr).detach()
            pred_fake = self.netD(imgs_sr)

            if self.ragan:
                pred_fake -= pred_real.mean(0, keepdim=True)

            # Adversarial loss
            loss_adv = self.criterion_GAN(pred_fake, True)
            # Calculate gradient penalty
            # gradient_penalty = self.gradient_penalty(imgs_hr, imgs_sr)

            g_loss = 0.3 * loss_edge + 0.7 * loss_pixel + self.alpha_adv * loss_adv #+ loss_content #+ 1 * gradient_penalty

            self.log('Step loss/generator', {'train_loss_edge': loss_edge,
                                             'train_loss_pixel': loss_pixel,
                                             # 'train_loss_content': loss_content,
                                             'train_loss_adv': loss_adv,
                                             # 'gradient_penalty': gradient_penalty,
                                             },
                     on_step=True,
                     on_epoch=False,
                     sync_dist=True,
                     prog_bar=True,
                     batch_size=self.batch_size)

            self.log('Epoch loss/generator', {'Train': g_loss,
                                              },
                     on_step=False,
                     on_epoch=True,
                     sync_dist=True,
                     batch_size=self.batch_size)

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

            self.log('Epoch loss/discriminator', {"Train": d_loss},
                     on_step=False,
                     on_epoch=True,
                     sync_dist=True,
                     batch_size=self.batch_size)

            return d_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            # ---------------------
            #  Validate Generator
            # ---------------------
            imgs_lr, imgs_hr = self.prepare_batch(batch)
            imgs_sr = self(imgs_lr)

            loss_pixel = self.criterion_pixel(imgs_sr, imgs_hr)
            loss_edge = self.criterion_edge(imgs_sr, imgs_hr)

            # gen_features = self.netF(torch.repeat_interleave(imgs_sr, 3, 1))
            # real_features = self.netF(torch.repeat_interleave(imgs_hr, 3, 1))
            # loss_content = self.criterion_content(gen_features, real_features)

            # Extract validity predictions from discriminator
            pred_real = self.netD(imgs_hr)
            pred_fake = self.netD(imgs_sr)

            # Relativistic average GAN
            if self.ragan:
                pred_real, pred_fake = pred_real - pred_fake.mean(0, keepdim=True), \
                                       pred_fake - pred_real.mean(0, keepdim=True)

            # Adversarial loss
            loss_adv = self.criterion_GAN(pred_fake, True)  # Gradient Penalty cannot be calculated during validation

            g_loss = 0.3 * loss_edge + 0.7 * loss_pixel + self.alpha_adv * loss_adv #+ loss_content

            # ---------------------
            #  Validate Discriminator
            # ---------------------

            # Adversarial loss for real and fake images
            loss_real = self.criterion_GAN(pred_real, True)
            loss_fake = self.criterion_GAN(pred_fake, False)

            d_loss = (loss_real + loss_fake) / 2

        self.log('Epoch loss/generator', {'Val': g_loss},
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 batch_size=self.batch_size)
        self.log('Epoch loss/discriminator', {"Val": d_loss},
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 batch_size=self.batch_size)

        if self.log_images_val:
            val_batches_done = batch_idx + self.current_epoch * self.val_len
            if val_batches_done % self.args.log_every_n_steps == 0:
                grid = self.imgs_cat(imgs_lr, imgs_hr, imgs_sr)
                self.logger.experiment.add_image('generated images/val',
                                                 make_grid(torch.clamp(grid, 0, 1), nrow=4),
                                                 val_batches_done,
                                                 dataformats='CHW')
                # save_dir = os.path.join(self.logger.log_dir, 'images', 'val')
                # os.makedirs(save_dir, exist_ok=True)
                # save_image(grid, save_dir + "/%04d.png" % val_batches_done, nrow=4, normalize=False)

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
