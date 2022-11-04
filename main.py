import os
from trainer_org import LitTrainer as LitTrainer_org
from trainer_gan import LitTrainer as LitTrainer_gan
from models.generator_ESRGAN import GeneratorRRDB as generator_ESRGAN
from models.generator_FSRCNN import FSRCNN as generator_FSRCNN
from models.generator_RRDB import GeneratorRRDB as generator_RRDB
from models.generator_DeepUResnet import DeepUResnet as generator_DeepUResnet
from models.discriminator import Discriminator
from models.feature_extractor import FeatureExtractor
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datetime import timedelta
from utils import print_config
from pytorch_lightning.loggers import WandbLogger
import wandb
import warnings

warnings.filterwarnings('ignore', '.*wandb run already in progress.*')

### Single config ###
default_config = {
    'optimizer': 'adam',
    'b1': 0.9,
    'b2': 0.5,
    'batch_size': 16,
    'num_filters': 64,
    'learning_rate_G': 2e-5,
    'learning_rate_D': 2e-5,
    'patch_size': 64,
    'alpha_pixel': 0.7,
    'alpha_edge': 0.3,
    'alpha_perceptual': 1,
    'alpha_adversarial': 0.1,
    'ragan': True,
    'gan_mode': 'vanilla',
    'edge_loss': 2,
    'netD_freq': 1,
    'data_resolution': '1mm_07mm',
    'nr_hcp_train': 30,
    'nr_sim_train': 30,
    'patch_overlap': 0.5,
    'generator': 'ESRGAN',
    'num_res_blocks': 1,
}

def main(default_config):
    pl.seed_everything(21011998)

    parser = ArgumentParser()
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--root_dir', default='/mnt/beta/djboonstoppel/Code', type=str)
    parser.add_argument('--warmup_batches', default=2500, type=int)
    parser.add_argument('--name', required=False, type=str)
    parser.add_argument('--wandb_project', default='test', type=str)
    parser.add_argument('--gan', action='store_true')
    parser.add_argument('--no_checkpointing', action='store_true')
    parser.set_defaults(gan=False)
    parser.set_defaults(no_checkpointing=False)

    log_folder = 'log/losses-final'

    # --precision=16 --gpus=1 --log_every_n_steps=50 --max_epochs=-1 --max_time="00:00:00:00"
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

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
    # print_config(config, args)

    if config.generator == 'ESRGAN':
        generator = generator_ESRGAN(channels=1, filters=config.num_filters, num_res_blocks=config.num_res_blocks)
    elif config.generator == 'RRDB':
        generator = generator_RRDB(channels=1, filters=config.num_filters, num_res_blocks=config.num_res_blocks)
    elif config.generator == 'DeepUResnet':
        generator = generator_DeepUResnet(nrfilters=config.num_filters)
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

    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.00,
                                        patience=3,
                                        verbose=False,
                                        mode='min',
                                        check_finite=True)

    early_stop_callback_if_nan = EarlyStopping(monitor='NCC_mean',
                                        min_delta=0.00,
                                        patience=25,
                                        verbose=False,
                                        mode='min',
                                        check_finite=True)

    checkpoint_callback_best = ModelCheckpoint(
        monitor="SSIM_mean",
        dirpath=os.path.join(args.root_dir, log_folder, wandb.run.name),
        filename=wandb.run.name + "-checkpoint-best",
        save_top_k=1,
        mode="max",
    )

    checkpoint_callback_time = ModelCheckpoint(
        dirpath=os.path.join(args.root_dir, log_folder, wandb.run.name),
        filename=wandb.run.name + "-checkpoint-{epoch}",
        save_top_k=-1,
        # train_time_interval=timedelta(minutes=2),
        every_n_epochs=1,
    )

    if args.no_checkpointing:
        callbacks = [lr_monitor, early_stop_callback, early_stop_callback_if_nan, checkpoint_callback_best]#, ModelPruning("l1_unstructured", amount=0.5)]
    else:
        callbacks = [lr_monitor, early_stop_callback, early_stop_callback_if_nan, checkpoint_callback_best, checkpoint_callback_time]#, ModelPruning("l1_unstructured", amount=0.5)]

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
        # strategy=DDPPlugin(find_unused_parameters=True),
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
