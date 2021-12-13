import os
import torch
import numpy as np
import torchio as tio
from trainer_gan import LitTrainer as LitTrainer_gan
from trainer_org import LitTrainer as LitTrainer_conventional
from generator import GeneratorRRDB
from discriminator import Discriminator
from feature_extractor import FeatureExtractor
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset_tio import data_split, Normalize, calculate_overlap

# print(os.getcwd())
# torch.cuda.empty_cache()

def main():
    pl.seed_everything(21011998)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--patients_frac', default=1.0, type=float)
    parser.add_argument('--std', default=0.3548, type=float)
    parser.add_argument('--patch_size', default=64, type=int)
    parser.add_argument('--patch_overlap', default=.5, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--root_dir', default='data', type=str)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitTrainer_conventional.add_model_specific_args(parser)
    args = parser.parse_args()

    generator = GeneratorRRDB(channels=1, filters=64, num_res_blocks=1)
    # discriminator = Discriminator(input_shape=(1, 64, 64))
    # feature_extractor = FeatureExtractor().to(device)

    logger = TensorBoardLogger('log', name='meuk1', default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="log/baseline",
        filename="baseline-{epoch:002d}",
        save_top_k=3,
        mode="min",
    )

    model = LitTrainer_conventional(netG=generator, args=args)

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=100,
        strategy='ddp_spawn',
        precision=args.precision,
        callbacks=[lr_monitor, checkpoint_callback],
    )

    trainer.fit(
        model,
    )


if __name__ == '__main__':
    main()

