import os
from trainer_org import LitTrainer as LitTrainer_org
from trainer_gan import LitTrainer as LitTrainer_gan
from models.generator_ESRGAN import GeneratorRRDB as generator_ESRGAN
from models.generator_FSRCNN import FSRCNN as generator_FSRCNN
from models.generator_RRDB import GeneratorRRDB as generator_RRDB
from models.generator_DeepUResnet import DeepUResnet as generator_DeepUResnet
from models.generator_DeepUResnet_v2 import build_deepunet as generator_DeepUResnet_v2
from models.discriminator import Discriminator
from models.feature_extractor import FeatureExtractor
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import warnings

warnings.filterwarnings('ignore', '.*wandb run already in progress.*')

### Config ###
default_config = {
    'optimizer': 'adam',  # optimizer
    'b1': 0.9,  # param for adam optimizer
    'b2': 0.5,  # param for adam optimizer
    'batch_size': 16,  # mini-batchsize for training
    'learning_rate_G': 2e-5,  # learning rate for generator
    'learning_rate_D': 2e-5,  # learning rate for generator
    'patch_size': 64,  # hxb for patch extraction

    # weightings of different loss components
    'alpha_pixel': 0.7,
    'alpha_edge': 0.3,
    'alpha_perceptual': 1,
    'alpha_adversarial': 0.01,  # 0.01 for WGAN, 0.1 for RaSGAN
    'alpha_gradientpenalty': 10,

    'generator': 'ESRGAN',  # generator architecture of choice
    'num_filters': 64,  # number of filters in generator layers (ESRGAN, RRDB, DeepUResnet)
    'num_res_blocks': 1,  # number or RRDB blocks in generator (ESRGAN, RRDB)

    'gan_mode': 'wgan',  # method to calculate adversarial loss 'wgan' for WGAN and 'vanilla' for RaSGAN
    'edge_loss': 2,  # method to calculate edge loss
    'netD_freq': 1,  # train disciminator n times for every time the generator is trained once

    # number of subjects for training and validation
    'nr_hcp_train': 30,
    'nr_sim_train': 30,
    'nr_hcp_val': 10,
    'nr_sim_val': 10,

    'patch_overlap': 0.5,  # only needed when grid sampler is used, can be ommitted for label sampler
}


def main(default_config):
    pl.seed_everything(21011998)

    parser = ArgumentParser()
    parser.add_argument('--num_workers', default=4, type=int)  # num workers for dataloading
    parser.add_argument('--root_dir', default='/mnt/beta/djboonstoppel/Code', type=str)  # root dir for code
    parser.add_argument('--warmup_batches', default=2500, type=int)  # number of batches only using pixel loss
    parser.add_argument('--name', required=False, type=str)  # name for logging, if empty, automatic name is generated
    parser.add_argument('--wandb_project', default='example', type=str)  # name for WandB project
    parser.add_argument('--gan', action='store_true')  # if flag is present, use adversarial approach
    parser.add_argument('--no_checkpointing',
                        action='store_true')  # if flag is present, only save best checkpoint, otherwise save every checkpoint
    parser.set_defaults(gan=False)
    parser.set_defaults(no_checkpointing=False)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    log_folder = os.path.join('log', args.wandb_project)

    if args.name:
        os.makedirs(os.path.join(args.root_dir, log_folder, args.name), exist_ok=True)
        run = wandb.init(config=default_config,
                         project=args.wandb_project,
                         name=args.name,
                         dir=os.path.join(args.root_dir, log_folder, args.name)
                         )
    else:
        run = wandb.init(config=default_config,
                         project=args.wandb_project,
                         )
        os.makedirs(os.path.join(args.root_dir, log_folder, wandb.run.name), exist_ok=True)

    if args.gan:
        parser = LitTrainer_gan.add_model_specific_args(parser)
        args = parser.parse_args()
    else:
        parser = LitTrainer_org.add_model_specific_args(parser)
        args = parser.parse_args()

    config = run.config_static

    if config.generator == 'ESRGAN':
        generator = generator_ESRGAN(channels=1, filters=config.num_filters, num_res_blocks=config.num_res_blocks)
    elif config.generator == 'RRDB':
        generator = generator_RRDB(channels=1, filters=config.num_filters, num_res_blocks=config.num_res_blocks)
    elif config.generator == 'DeepUResnet':
        generator = generator_DeepUResnet(nrfilters=config.num_filters)
    elif config.generator == 'DeepUResnet_v2':
        generator = generator_DeepUResnet_v2()
    elif config.generator == 'FSRCNN':
        generator = generator_FSRCNN(scale_factor=1)
    else:
        raise NotImplementedError(
            "Generator architecture '{}' is not recognized or implemented".format(config.generator))

    discriminator = Discriminator(input_shape=(1, config.patch_size, config.patch_size))
    feature_extractor = FeatureExtractor()

    logger = WandbLogger(project=args.wandb_project,
                         name=wandb.run.name,
                         save_dir=os.path.join(args.root_dir, log_folder, wandb.run.name),
                         log_model=False, )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # stop training if not improving
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.00,
                                        patience=5,
                                        verbose=False,
                                        mode='min',
                                        check_finite=True)

    # stop training if NaN occurs
    early_stop_callback_if_nan = EarlyStopping(monitor='NCC_mean',
                                               min_delta=0.00,
                                               patience=25,
                                               verbose=False,
                                               mode='min',
                                               check_finite=True)

    # save best checkpoint
    checkpoint_callback_best = ModelCheckpoint(
        monitor="SSIM_mean",
        dirpath=os.path.join(args.root_dir, log_folder, wandb.run.name),
        filename=wandb.run.name + "-checkpoint-best",
        save_top_k=1,
        mode="max",
    )

    # save every checkpoint
    checkpoint_callback_time = ModelCheckpoint(
        dirpath=os.path.join(args.root_dir, log_folder, wandb.run.name),
        filename=wandb.run.name + "-checkpoint-{epoch}",
        save_top_k=-1,
        every_n_epochs=1,
    )

    if args.no_checkpointing:
        callbacks = [lr_monitor, early_stop_callback, early_stop_callback_if_nan,
                     checkpoint_callback_best]
    else:
        callbacks = [lr_monitor, early_stop_callback, early_stop_callback_if_nan, checkpoint_callback_best,
                     checkpoint_callback_time]

    if args.gan:
        model = LitTrainer_gan(netG=generator, netF=feature_extractor, netD=discriminator, args=args, config=config)
    else:
        model = LitTrainer_org(netG=generator, netF=feature_extractor, args=args, config=config)

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        max_time=args.max_time,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        precision=args.precision,
        callbacks=callbacks,
        enable_progress_bar=True,
        num_sanity_val_steps=args.num_sanity_val_steps,
        val_check_interval=args.val_check_interval,
    )

    trainer.fit(
        model,
    )


if __name__ == '__main__':
    main(default_config)
