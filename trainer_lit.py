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

class LitTrainer(pl.LightningModule):
    def __init__(self,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 # generator, discriminator, training_loader, validation_loader,
                 # feature_extractor, output_dir, std,
                 # batch_size=8,
                 # lr=0.0002,
                 # b1=0.9,
                 # b2=0.999,
                 # start_epoch=1,
                 # n_epochs=10,
                 # lambda_content=1,
                 # lambda_edge=0.3,
                 # lambda_adv=5e-3,
                 # lambda_pixel=1e-2,
                 # sample_interval=100,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        # self.netG = generator.to(self.device)
        self.netG = GeneratorRRDB(channels=1, filters=64, num_res_blocks=1)
        # self.netD = discriminator.to(self.device)
        self.netD = Discriminator(input_shape=(1, 64, 64))
        # self.netF = feature_extractor.to(self.device).eval()

        # self.start_epoch = start_epoch
        # if start_epoch != 1:
        #     self.load(os.path.join(output_dir, 'saved_models', '%02d.model' % start_epoch))

        # self.criterion_GAN = torch.nn.BCEWithLogitsLoss().to(self.device)
        # self.criterion_content = torch.nn.L1Loss().to(self.device)
        # self.criterion_pixel = torch.nn.L1Loss().to(self.device)

        # self.lambda_adv = lambda_adv
        # self.lambda_pixel = lambda_pixel
        # self.lambda_content = lambda_content
        # self.lambda_edge = lambda_edge
        # self.sample_interval = sample_interval
        # self.epochs = n_epochs
        # self.std = std

        # self.training_loader = training_loader
        # self.validation_loader = validation_loader
        # self.batch_size = batch_size
        # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(b1, b2))
        # self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(b1, b2))
        # self.metric = {
        #     'train_loss_G': [],
        #     'train_loss_content_G': [],
        #     'train_loss_adversarial_G': [],
        #     'train_loss_pixel_G': [],
        #     'train_loss_edge_G': [],
        #     'train_loss_D': [],
        #     'val_loss_G': [],
        #     'val_loss_content_G': [],
        #     'val_loss_adversarial_G': [],
        #     'val_loss_pixel_G': [],
        #     'val_loss_edge_G': [],
        #     'val_loss_D': [],
        # }
        # self.scores = {
        #     'SSIM': [],
        #     'NCC': [],
        #     'NRSME': [],
        # }

        # self.output_dir = output_dir
        # os.makedirs(output_dir + '/images/train', exist_ok=True)
        # os.makedirs(output_dir + '/images/val', exist_ok=True)
        # os.makedirs(output_dir + '/saved_models', exist_ok=True)
        # self.writer = SummaryWriter(output_dir)

    def forward(self, inputs):
        return self.netG(inputs)

    def adversarial_loss(self, gen, target, valid):
        # torch.nn.BCEWithLogitsLoss()(gen - target.mean(0, keepdim=True), valid)

        return torch.nn.BCEWithLogitsLoss()(gen - target.mean(0, keepdim=True), valid)

    def prepare_batch(self, batch):
        return batch['LR'][tio.DATA].squeeze(4), batch['HR'][tio.DATA].squeeze(4)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs_lr, imgs_hr = self.prepare_batch(batch)

        # Adversarial ground truths
        valid = torch.ones((imgs_lr.size(0), *self.netD.output_shape))
        valid = valid.type_as(imgs_lr)
        fake = torch.zeros((imgs_lr.size(0), *self.netD.output_shape))
        fake = fake.type_as(imgs_lr)
        # valid = Variable(self.Tensor(np.ones((imgs_lr.size(0), *self.netD.output_shape))), requires_grad=False)
        # fake = Variable(self.Tensor(np.zeros((imgs_lr.size(0), *self.netD.output_shape))), requires_grad=False)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.gen_hr = self(imgs_lr)

            # # log sampled images
            # sample_imgs = self.generated_imgs[:6]
            # grid = torchvision.utils.make_grid(sample_imgs)
            # self.logger.experiment.add_image("generated_images", grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            # valid = torch.ones(imgs_lr.size(0), 1)
            # valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.netD(self.gen_hr), self.netD(imgs_hr).detach(), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            # valid = torch.ones(imgs.size(0), 1)
            # valid = valid.type_as(imgs)

            # real_loss = self.adversarial_loss(self.netD(imgs_lr), valid)

            # how well can it label as fake?
            # fake = torch.zeros(imgs.size(0), 1)
            # fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.netD(self.gen_hr), self.netD(imgs_hr).detach(), fake)
            real_loss = self.adversarial_loss(self.netD(imgs_hr), self.netD(self.gen_hr).detach(), valid)

            # discriminator loss is the average of these
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


