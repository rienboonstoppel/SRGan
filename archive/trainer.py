import os
import torchvision
from archive.dataset import *
from metrics import *
from torch.autograd import Variable
import sys
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from edgeloss import *


class Trainer:
    def __init__(self, generator, discriminator, training_loader, validation_loader,
                 feature_extractor, output_dir, std,
                 batch_size=8,
                 lr=0.0002,
                 b1=0.9,
                 b2=0.999,
                 start_epoch=1,
                 n_epochs=10,
                 lambda_content=1,
                 lambda_edge=0.3,
                 lambda_adv=5e-3,
                 lambda_pixel=1e-2,
                 sample_interval=100,
                 ):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.netG = generator.to(self.device)
        self.netD = discriminator.to(self.device)
        self.netF = feature_extractor.to(self.device).eval()

        self.start_epoch = start_epoch
        if start_epoch != 1:
            self.load(os.path.join(output_dir, 'saved_models', '%02d.model' % start_epoch))

        self.criterion_GAN = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.criterion_content = torch.nn.L1Loss().to(self.device)
        self.criterion_pixel = torch.nn.L1Loss().to(self.device)

        self.lambda_adv = lambda_adv
        self.lambda_pixel = lambda_pixel
        self.lambda_content = lambda_content
        self.lambda_edge = lambda_edge
        self.sample_interval = sample_interval
        self.epochs = n_epochs
        self.std = std

        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.batch_size = batch_size
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(b1, b2))
        self.metric = {
            'train_loss_G': [],
            'train_loss_content_G': [],
            'train_loss_adversarial_G': [],
            'train_loss_pixel_G': [],
            'train_loss_edge_G': [],
            'train_loss_D': [],
            'val_loss_G': [],
            'val_loss_content_G': [],
            'val_loss_adversarial_G': [],
            'val_loss_pixel_G': [],
            'val_loss_edge_G': [],
            'val_loss_D': [],
        }
        self.scores = {
            'SSIM': [],
            'NCC': [],
            'NRSME': [],
        }

        self.output_dir = output_dir
        os.makedirs(output_dir + '/images/train', exist_ok=True)
        os.makedirs(output_dir + '/images/val', exist_ok=True)
        os.makedirs(output_dir + '/saved_models', exist_ok=True)
        self.writer = SummaryWriter(output_dir)

    def make_grid(self, batch, gen, output_dir, name, batches_done):
        img_grid = torch.cat(((batch['LR'] * self.std).to(self.device),
                              (batch['HR'] * self.std).to(self.device),
                              torch.clamp((gen * self.std), 0, 1),
                              (batch['LR'] * self.std).to(self.device) - (gen * self.std) + .5), -1)
        path = os.path.join(output_dir, 'images', name, '%05d.png' % batches_done)
        save_image(img_grid, path, nrow=1, normalize=False)
        tb_grid = torchvision.utils.make_grid(img_grid, nrow=1)
        return tb_grid

    def save(self, epoch):
        fname = os.path.join(self.output_dir, 'saved_models', '%02d.model' % epoch)
        torch.save({'generator': self.netG.state_dict(),
                    'discriminator': self.netD.state_dict(),
                    'epoch': epoch}, fname)

    def load(self, fname):
        state_dict = torch.load(fname)
        self.netG.load_state_dict(state_dict['generator'])
        self.netD.load_state_dict(state_dict['discriminator'])
        print('Model loaded, trained for {} epochs'.format(state_dict['epoch']))

    def training(self, train_batch):
        self.netG.train()
        self.netD.train()
        imgs_lr = Variable(train_batch['LR'].type(self.Tensor))
        imgs_hr = Variable(train_batch['HR'].type(self.Tensor))

        # Adversarial ground truths
        valid = Variable(self.Tensor(np.ones((imgs_lr.size(0), *self.netD.output_shape))), requires_grad=False)
        fake = Variable(self.Tensor(np.zeros((imgs_lr.size(0), *self.netD.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        self.optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = self.netG(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)

        # Extract validity predictions from discriminator
        pred_real = self.netD(imgs_hr).detach()
        pred_fake = self.netD(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        gen_features = self.netF(torch.repeat_interleave(gen_hr, 3, 1))
        real_features = self.netF(torch.repeat_interleave(imgs_hr, 3, 1)).detach()
        loss_content = self.criterion_content(gen_features, real_features)

        # Edge loss
        loss_edge = edge_loss1(gen_hr, imgs_hr)

        # Total generator loss
        loss_G = self.lambda_content * loss_content + \
                 self.lambda_adv * loss_GAN + \
                 self.lambda_pixel * loss_pixel + \
                 self.lambda_edge * loss_edge
        # loss_G = .7 * loss_pixel + .3 * loss_edge

        loss_G.backward()
        self.optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_D.zero_grad()

        pred_real = self.netD(imgs_hr)
        pred_fake = self.netD(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        self.optimizer_D.step()

        self.metric['train_loss_G'].append(loss_G.item())
        self.metric['train_loss_content_G'].append(loss_content.item())
        self.metric['train_loss_adversarial_G'].append(loss_GAN.item())
        self.metric['train_loss_pixel_G'].append(loss_pixel.item())
        self.metric['train_loss_edge_G'].append(loss_edge.item())
        self.metric['train_loss_D'].append(loss_D.item())
        return gen_hr

    @torch.no_grad()
    def validate(self, validation_batch):
        self.netG.eval()
        self.netD.eval()
        imgs_lr = Variable(validation_batch['LR'].type(self.Tensor))
        imgs_hr = Variable(validation_batch['HR'].type(self.Tensor))

        # Adversarial ground truths
        valid = Variable(self.Tensor(np.ones((imgs_lr.size(0), *self.netD.output_shape))), requires_grad=False)
        fake = Variable(self.Tensor(np.zeros((imgs_lr.size(0), *self.netD.output_shape))), requires_grad=False)

        # ------------------
        #  Validate Generators
        # ------------------

        # Generate a high resolution image from low resolution input
        gen_hr = self.netG(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)

        # Extract validity predictions from discriminator
        pred_real = self.netD(imgs_hr).detach()
        pred_fake = self.netD(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        gen_features = self.netF(torch.repeat_interleave(gen_hr, 3, 1))
        real_features = self.netF(torch.repeat_interleave(imgs_hr, 3, 1)).detach()
        loss_content = self.criterion_content(gen_features, real_features)

        # Edge loss
        loss_edge = edge_loss1(gen_hr, imgs_hr)

        # Total generator loss
        loss_G = self.lambda_content * loss_content + \
                 self.lambda_adv * loss_GAN + \
                 self.lambda_pixel * loss_pixel + \
                 self.lambda_edge * loss_edge

        # ---------------------
        #  Validate Discriminator
        # ---------------------

        pred_real = self.netD(imgs_hr)
        pred_fake = self.netD(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        self.metric['val_loss_G'].append(loss_G.item())
        self.metric['val_loss_content_G'].append(loss_content.item())
        self.metric['val_loss_adversarial_G'].append(loss_GAN.item())
        self.metric['val_loss_pixel_G'].append(loss_pixel.item())
        self.metric['val_loss_edge_G'].append(loss_pixel.item())
        self.metric['val_loss_D'].append(loss_D.item())

        ncc, ssim, nrsme = get_scores_batch(imgs_hr.cpu(), gen_hr.detach().cpu())
        self.scores['NCC'].append(ncc)
        self.scores['SSIM'].append(ssim)
        self.scores['NRSME'].append(nrsme)

        return gen_hr

    def fit(self):
        training_loader = self.training_loader
        validation_loader = self.validation_loader
        sys.stdout.flush()
        for epoch in range(self.start_epoch, self.epochs + 1):
            print('Epoch %d' % epoch)
            with tqdm(desc='Training  ', total=len(training_loader),
                      bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}') as pbar:
                for i, training_batch in enumerate(training_loader):
                    batches_done = (epoch - 1) * len(training_loader) + i
                    gen_hr_train = self.training(training_batch)

                    self.writer.add_scalars(
                        'Generator batch loss', {'Train': self.metric['train_loss_G'][-1],
                                                 'Train_pixel': self.metric['train_loss_pixel_G'][-1],
                                                 'Train_adversarial': self.metric['train_loss_adversarial_G'][-1],
                                                 'Train_content': self.metric['train_loss_content_G'][-1],
                                                 'Train_edge': self.metric['train_loss_edge_G'][-1],
                                                 }, batches_done)

                    self.writer.add_scalars(
                        'Discriminator batch loss', {'Train': self.metric['train_loss_D'][-1],
                                                     }, batches_done)

                    if batches_done % self.sample_interval == 0:
                        train_grid = self.make_grid(training_batch, gen_hr_train,
                                                    self.output_dir, 'train', batches_done)
                        self.writer.add_image('Training_batches', train_grid, batches_done, dataformats='CHW')

                    it_metrics = {
                        "loss_G": '%1.3f' % self.metric["train_loss_G"][-1],
                        "loss_D": '%1.3f' % self.metric["train_loss_D"][-1],
                    }

                    pbar.set_postfix(**it_metrics)
                    pbar.update()

            with tqdm(desc='Validation', total=len(validation_loader),
                      bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}') as pbar:
                for i, validation_batch in enumerate(validation_loader):
                    batches_done = (epoch - 1) * len(validation_loader) + i
                    gen_hr_val = self.validate(validation_batch)

                    self.writer.add_scalars(
                        'Validation batch scores', {'NCC': self.scores['NCC'][-1],
                                                    'SSIM': self.scores['SSIM'][-1],
                                                    'NRSME': self.scores['NRSME'][-1],
                                                    }, batches_done)

                    if batches_done % self.sample_interval == 0:
                        train_grid = self.make_grid(validation_batch, gen_hr_val,
                                                    self.output_dir, 'val', batches_done)
                        self.writer.add_image('Validation_batches', train_grid, batches_done, dataformats='CHW')

                    it_metrics = {
                        "loss_G": '%1.3f' % self.metric["val_loss_G"][-1],
                        "loss_D": '%1.3f' % self.metric["val_loss_D"][-1],
                    }

                    pbar.set_postfix(**it_metrics)
                    pbar.update()

            train_epoch_loss_G = np.mean(self.metric['train_loss_G'][-len(training_loader):])
            val_epoch_loss_G = np.mean(self.metric['val_loss_G'][-len(validation_loader):])
            train_epoch_loss_D = np.mean(self.metric['train_loss_D'][-len(training_loader):])
            val_epoch_loss_D = np.mean(self.metric['val_loss_D'][-len(validation_loader):])

            self.writer.add_scalars(
                'Generator epoch loss', {'Train': train_epoch_loss_G,
                                         'Val': val_epoch_loss_G,
                                         }, epoch)
            self.writer.add_scalars(
                'Discriminator epoch loss', {'Train': train_epoch_loss_D,
                                             'Val': val_epoch_loss_D,
                                             }, epoch)

            self.save(epoch)

        sys.stdout.flush()
        self.writer.close()

        return self.metric
