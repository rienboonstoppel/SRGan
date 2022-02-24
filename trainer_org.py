import os
import pytorch_lightning as pl
import torch
import torchio as tio
import torchvision
from dataset_tio import data_split, Normalize, calculate_overlap
from edgeloss import edge_loss1, edge_loss2, edge_loss3
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
                 netG, netF,
                 config,
                 args,
                 **kwargs
                 ):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=['netG', 'netF', 'netD'])

        self.netG = netG
        self.netF = netF.eval()

        self.criterion_pixel = torch.nn.L1Loss()    # method to calculate pixel differences
        self.criterion_content = torch.nn.L1Loss()  # method to calculate differences between vgg features
        self.criterion_edge = globals()['edge_loss' + str(config['edge_loss'])]

        self.patients_frac = config['patients_frac']
        self.patch_overlap = config['patch_overlap']
        self.batch_size = config['batch_size']
        self.patch_size = config['patch_size']
        self.alpha_content = config['alpha_content']
        self.log_images = True

        if config['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(netG.parameters(),
                                             lr=config['learning_rate'],
                                             momentum=0.9,
                                             nesterov=True)
        elif config['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(netG.parameters(),
                                              lr=config['learning_rate'],
                                              betas=(config['b1'], config['b2']))


    def make_grid(self, imgs_lr, imgs_hr, imgs_sr):
        imgs_lr = torch.clamp((imgs_lr[:10] * self.args.std), 0, 1).squeeze()
        imgs_hr = torch.clamp((imgs_hr[:10] * self.args.std), 0, 1).squeeze()
        imgs_sr = torch.clamp((imgs_sr[:10] * self.args.std), 0, 1).squeeze()
        diff = (imgs_hr - imgs_sr) * 2 + .5

        img_grid = torch.cat(
            [torch.stack([a, b, c, d]) for a, b, c, d in zip(imgs_lr, imgs_hr, imgs_sr, diff)]).unsqueeze(1)

        tb_grid = torchvision.utils.make_grid(img_grid, nrow=4)
        return tb_grid

    def forward(self, inputs):
        return self.netG(inputs)

    def prepare_batch(self, batch):
        return batch['LR'][tio.DATA].squeeze(4), batch['HR'][tio.DATA].squeeze(4)

    def training_step(self, batch, batch_idx):
        imgs_lr, imgs_hr = self.prepare_batch(batch)
        imgs_sr = self(imgs_lr)

        loss_edge = self.criterion_edge(imgs_sr, imgs_hr)
        loss_pixel = self.criterion_pixel(imgs_sr, imgs_hr)

        gen_features = self.netF(torch.repeat_interleave(imgs_sr, 3, 1))
        real_features = self.netF(torch.repeat_interleave(imgs_hr, 3, 1)).detach()
        loss_content = self.criterion_content(gen_features, real_features)

        g_loss = 0.3 * loss_edge + 0.7 * loss_pixel + self.alpha_content * loss_content

        self.log('Step loss/generator', {'train_loss_edge': loss_edge,
                                         'train_loss_pixel': loss_pixel,
                                         'train_loss_content': loss_content,
                                         }, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True,
                 batch_size=self.batch_size)

        self.log('Epoch loss/generator', {'Train': g_loss,
                                          }, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        if self.log_images:
            train_batches_done = batch_idx + self.current_epoch * self.train_len
            if train_batches_done % (self.args.log_every_n_steps * 5) == 0:
                grid = self.make_grid(imgs_lr, imgs_hr, imgs_sr)
                self.logger.experiment.add_image('generated images/train', grid, train_batches_done,
                                                 dataformats='CHW')

        return g_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs_lr, imgs_hr = self.prepare_batch(batch)
            imgs_sr = self(imgs_lr)
            loss_edge = self.criterion_edge(imgs_sr, imgs_hr)
            loss_pixel = self.criterion_pixel(imgs_sr, imgs_hr)

            gen_features = self.netF(torch.repeat_interleave(imgs_sr, 3, 1))
            real_features = self.netF(torch.repeat_interleave(imgs_hr, 3, 1)).detach()
            loss_content = self.criterion_content(gen_features, real_features)

            g_loss = 0.3 * loss_edge + 0.7 * loss_pixel + self.alpha_content * loss_content

            self.log('Epoch loss/generator', {'Val': g_loss}, on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=self.batch_size)

            if self.log_images:
                val_batches_done = batch_idx + self.current_epoch * self.val_len
                if val_batches_done % self.args.log_every_n_steps == 0:
                    grid = self.make_grid(imgs_lr, imgs_hr, imgs_sr)
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
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99999)}
            }