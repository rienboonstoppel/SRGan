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
from skimage.metrics import structural_similarity as SSIM

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
        self.criterion_GAN = GANLoss(gan_mode=config['gan_mode'])  # method to calculate adversarial loss
        # self.criterion_GAN = GANLoss(gan_mode='vanilla')  # method to calculate adversarial loss
        self.gradient_penalty = GradientPenalty(critic=self.netD, fake_label=1.0)
        self.criterion_edge = globals()['edge_loss' + str(config['edge_loss'])]
        self.alpha_adv = config['alpha_adversarial']
        self.netD_freq = config['netD_freq']

        self.datasource = config['datasource']
        self.patients_frac = config['patients_frac']
        self.patch_overlap = config['patch_overlap']
        self.batch_size = config['batch_size']
        self.patch_size = config['patch_size']
        self.ragan = config['ragan']
        self.log_images_train = False
        self.log_images_val = True

        if config['optimizer'] == 'sgd':
            self.optimizer_G = torch.optim.SGD(netG.parameters(),
                                             lr=config['learning_rate_G'],
                                             momentum=0.9,
                                             nesterov=True)
        elif config['optimizer'] == 'adam':
            self.optimizer_G = torch.optim.Adam(netG.parameters(),
                                              lr=config['learning_rate_G'],
                                              betas=(config['b1'], config['b2']))
        if config['optimizer'] == 'sgd':
            self.optimizer_D = torch.optim.SGD(netD.parameters(),
                                             lr=config['learning_rate_D'],
                                             momentum=0.9,
                                             nesterov=True)
        elif config['optimizer'] == 'adam':
            self.optimizer_D = torch.optim.Adam(netD.parameters(),
                                              lr=config['learning_rate_D'],
                                              betas=(config['b1'], config['b2']))

    def imgs_cat(self, imgs_lr, imgs_hr, imgs_sr):
        imgs_lr = (imgs_lr[:10] * self.args.std).squeeze()
        imgs_hr = (imgs_hr[:10] * self.args.std).squeeze()
        imgs_sr = (imgs_sr[:10] * self.args.std).squeeze()
        diff = (imgs_hr - imgs_sr) * 2 + .5
        img_grid = torch.cat(
            [torch.stack([a, b, c, d]) for a, b, c, d in zip(imgs_lr, imgs_hr, imgs_sr, diff)]).unsqueeze(1)
        return img_grid

    def post_proc(self, img: torch.Tensor, bg_idx: np.ndarray, crop_coords: tuple) -> np.ndarray:
        img[bg_idx] = 0
        min, max = crop_coords
        img = img.squeeze(0)[min[0]:max[0] + 1, min[1]:max[1] + 1, min[2]:max[2] + 1].numpy() * self.args.std
        return img

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

            gen_features = self.netF(torch.repeat_interleave(imgs_sr, 3, 1))
            real_features = self.netF(torch.repeat_interleave(imgs_hr, 3, 1)).detach()
            loss_content = self.criterion_content(gen_features, real_features)

            # Extract validity predictions from discriminator
            pred_real = self.netD(imgs_hr).detach()
            pred_fake = self.netD(imgs_sr)

            if self.ragan:
                pred_fake -= pred_real.mean(0, keepdim=True)

            # Adversarial loss
            loss_adv = self.criterion_GAN(pred_fake, True)
            # Calculate gradient penalty
            # gradient_penalty = self.gradient_penalty(imgs_hr, imgs_sr)

            g_loss = 0.3 * loss_edge + 0.7 * loss_pixel + self.alpha_adv * loss_adv + loss_content #+ 1 * gradient_penalty

            self.log('Step loss/generator', {'train_loss_edge': loss_edge,
                                             'train_loss_pixel': loss_pixel,
                                             'train_loss_content': loss_content,
                                             'train_loss_adv': loss_adv,
                                             # 'gradient_penalty': gradient_penalty,
                                             },
                     on_step=True,
                     on_epoch=False,
                     sync_dist=True,
                     prog_bar=False,
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

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            # ---------------------
            #  Validate Generator
            # ---------------------
            imgs_lr, imgs_hr = self.prepare_batch(batch)
            imgs_sr = self(imgs_lr)

            loss_pixel = self.criterion_pixel(imgs_sr, imgs_hr)
            loss_edge = self.criterion_edge(imgs_sr, imgs_hr)

            gen_features = self.netF(imgs_sr.repeat(1, 3, 1 ,1))
            real_features = self.netF(imgs_hr.repeat(1, 3, 1 ,1))
            loss_content = self.criterion_content(gen_features, real_features)

            # Extract validity predictions from discriminator
            pred_real = self.netD(imgs_hr)
            pred_fake = self.netD(imgs_sr)

            # Relativistic average GAN
            if self.ragan:
                pred_real, pred_fake = pred_real - pred_fake.mean(0, keepdim=True), \
                                       pred_fake - pred_real.mean(0, keepdim=True)

            # Adversarial loss
            loss_adv = self.criterion_GAN(pred_fake, True)  # Gradient Penalty cannot be calculated during validation

            g_loss = 0.3 * loss_edge + 0.7 * loss_pixel + self.alpha_adv * loss_adv + loss_content

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
        if dataloader_idx > 0:
            imgs_lr, imgs_hr = self.prepare_batch(batch)
            locations = batch[tio.LOCATION]
            imgs_sr = self(imgs_lr)
            return imgs_hr, imgs_sr, locations


    def validation_epoch_end(self, outputs):
        val_loss = outputs[0]
        avg_loss = torch.mean(torch.stack(val_loss))
        self.log('val_loss', avg_loss, sync_dist=True)

        data = outputs[1:]
        # print(len(data))



        SSIMs = []
        # HR_aggs = []
        # SR_aggs = []
        for i in range(len(data)):
            for j in range(len(data[i])):
                imgs_hr, imgs_sr, locations = data[i][j]
                self.aggregator_HR.add_batch(imgs_hr.unsqueeze(4), locations)
                self.aggregator_SR.add_batch(imgs_sr.unsqueeze(4), locations)

            HR_agg = self.aggregator_HR.get_output_tensor()
            SR_agg = self.aggregator_SR.get_output_tensor()
            # HR_aggs.append(HR_agg*self.args.std)
            # SR_aggs.append(SR_agg*self.args.std)
            bg_idx, brain_idx = self.post_proc_info[i]
            # print(HR_agg.shape, bg_idx, brain_idx)
            # print(SR_agg.shape, bg_idx, brain_idx)
            HR_agg = self.post_proc(HR_agg, bg_idx, brain_idx)
            SR_agg = self.post_proc(SR_agg, bg_idx, brain_idx)
            #
            ssim = SSIM(HR_agg, SR_agg, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            SSIMs.append(ssim)
            # print(ssim)
        self.log('SSIM', np.mean(SSIMs))

        # self.logger.experiment.add_images('generated images/val_aggregated',
        #                                  torch.clamp(torch.cat(SR_aggs, dim=3), 0, 1),
        #                                  self.current_epoch,
        #                                  dataformats='CHWN')

    def setup(self, stage='fit'):
        args = self.args
        data_path = os.path.join(args.root_dir, 'data')
        train_subjects = data_split('training',
                                    patients_frac=self.patients_frac,
                                    root_dir=data_path,
                                    datasource=self.datasource,
                                    numslices=None)
        val_subjects = data_split('validation',
                                  patients_frac=self.patients_frac,
                                  root_dir=data_path,
                                  datasource=self.datasource,
                                  numslices=None)
        test_subjects = data_split('test',
                                   patients_frac=self.patients_frac,
                                   datasource=self.datasource,
                                   numslices=50)

        self.post_proc_info = []
        for subject in test_subjects:
            mask = subject['HR_msk'].data

            bg_idx = np.where(mask == 0)
            brain_idx = np.where(mask.squeeze(0) != 0)
            crop_coords = ([brain_idx[i].min() for i in range(len(brain_idx))],
                           [brain_idx[i].max() for i in range(len(brain_idx))])
            self.post_proc_info.append((bg_idx, crop_coords))
        print('post_proc_len', len(self.post_proc_info))

        training_transform = tio.Compose([
            Normalize(std=args.std),
            # tio.RandomNoise(p=0.5),
            tio.RandomFlip(axes=(0, 1), flip_probability=0.5),
            tio.RandomFlip(axes=(0, 1), flip_probability=0.75),
            # tio.RandomAffine(degrees=(0, 0, 0, 0, 0, 360),
            #                  default_pad_value=0,
            #                  scales=0,
            #                  translation=0,
            #                  isotropic=True
            #                  )
        ])

        test_transform = tio.Compose([
            Normalize(std=args.std),
        ])

        self.training_set = tio.SubjectsDataset(
            train_subjects, transform=training_transform)

        self.val_set = tio.SubjectsDataset(
            val_subjects, transform=training_transform)

        self.test_set = tio.SubjectsDataset(
            test_subjects, transform=test_transform)

        self.overlap, nr_patches = calculate_overlap(train_subjects[0]['LR'],
                                                (self.patch_size, self.patch_size),
                                                (self.patch_overlap, self.patch_overlap)
                                                )
        self.samples_per_volume = nr_patches

        # self.sampler = tio.data.GridSampler(patch_size=(self.patch_size, self.patch_size, 1),
        #                                     patch_overlap=overlap,
        #                                     # padding_mode=0,
        #                                     )

        probabilities = {0: 0, 1: 1}
        # , 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1,
        #                  14: 1, 15: 1, 16: 1}

        self.sampler = tio.data.LabelSampler(
            patch_size=(self.patch_size, self.patch_size, 1),
            label_name='HR_msk_bin',
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
        for i in range(15):
            grid_sampler = tio.inference.GridSampler(
                subject=self.test_set[i],
                patch_size=(self.patch_size, self.patch_size, 1),
                patch_overlap=self.overlap,
                padding_mode=0,
            )
            test_loader = torch.utils.data.DataLoader(
                grid_sampler, batch_size=self.batch_size)
            loaders.append(test_loader)

        # grid_sampler_0 = tio.inference.GridSampler(
        #     subject=self.test_set[0],
        #     patch_size=(self.patch_size, self.patch_size, 1),
        #     patch_overlap=self.overlap,
        #     padding_mode=0,
        # )
        #
        # grid_sampler_1 = tio.inference.GridSampler(
        #     subject=self.test_set[1],
        #     patch_size=(self.patch_size, self.patch_size, 1),
        #     patch_overlap=self.overlap,
        #     padding_mode=0,
        # )

        self.aggregator_HR=tio.inference.GridAggregator(grid_sampler)
        self.aggregator_SR=tio.inference.GridAggregator(grid_sampler)

        # test_loader_0 = torch.utils.data.DataLoader(
        #         grid_sampler_0, batch_size=self.batch_size)
        # test_loader_1 = torch.utils.data.DataLoader(
        #         grid_sampler_1, batch_size=self.batch_size)
        self.test_len = len(self.test_set)

        return [val_loader] + loaders

    # def test_dataloader(self):
    #     test_queue = tio.Queue(
    #         subjects_dataset=self.test_set,
    #         max_length=self.samples_per_volume * 5,
    #         samples_per_volume=self.samples_per_volume,
    #         sampler=self.sampler,
    #         num_workers=self.args.num_workers,
    #         shuffle_subjects=False,
    #         shuffle_patches=False,
    #     )
    #     test_loader = torch.utils.data.DataLoader(
    #         test_queue,
    #         batch_size=self.batch_size,
    #         num_workers=0,
    #     )
    #     self.test_len = len(test_loader)
    #     return test_loader

    # def test_dataloader(self):
    #     test_loaders = []
    #     self.test_aggregators = []
    #     for i in range(len(self.test_set)):
    #         grid_sampler = tio.inference.GridSampler(
    #             subject=self.test_set[i],
    #             patch_size=(self.patch_size, self.patch_size, 1),
    #             patch_overlap=self.overlap,
    #             padding_mode=0,
    #         )
    #         self.test_aggregators.append(tio.inference.GridAggregator(grid_sampler))
    #         test_loaders.append(torch.utils.data.DataLoader(
    #             grid_sampler, batch_size=self.batch_size))
    #
    #
    #     self.test_len = len(self.test_set)
    #     return test_loaders

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
