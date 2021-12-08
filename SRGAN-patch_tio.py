import os
import torch
# import numpy as np
import matplotlib.pyplot as plt
import torchio as tio
import torchvision
from trainer_gan import LitTrainer as LitTrainer_gan
from trainer import LitTrainer as LitTrainer_conventional
from generator import GeneratorRRDB
from discriminator import Discriminator
from feature_extractor import FeatureExtractor
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from collections import OrderedDict
import argparse
from collections import namedtuple
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg
from dataset_tio import ImagePair, data_split, Normalize, calculate_overlap

# print(os.getcwd())
# torch.cuda.empty_cache()


def main():
    train_subjects = data_split('training', patients_frac=1)
    val_subjects = data_split('validation', patients_frac=1)

    std = 0.3548

    training_transform = tio.Compose([
        Normalize(std=std),
        # tio.RandomNoise(p=0.5),
        tio.RandomFlip(),
    ])

    training_set = tio.SubjectsDataset(
        train_subjects, transform=training_transform)

    val_set = tio.SubjectsDataset(
        val_subjects, transform=training_transform)

    batch_size = 256
    training_batch_size = batch_size
    validation_batch_size = batch_size

    num_workers = 4
    patch_size = (64, 64)
    ovl_perc = (.5, .5)
    overlap, nr_patches = calculate_overlap(train_subjects[0]['LR'], patch_size, ovl_perc)
    samples_per_volume = nr_patches

    max_queue_length = samples_per_volume*100
    sampler = tio.data.GridSampler(patch_size=(*patch_size,1), patch_overlap=overlap)#, padding_mode=0)

    training_queue = tio.Queue(
        subjects_dataset=training_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    val_queue = tio.Queue(
        subjects_dataset=val_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    training_loader = torch.utils.data.DataLoader(
        training_queue, batch_size=training_batch_size)

    val_loader = torch.utils.data.DataLoader(
        val_queue, batch_size=validation_batch_size)

    generator = GeneratorRRDB(channels=1, filters=64, num_res_blocks=1)
    # discriminator = Discriminator(input_shape=(1, 64, 64))
    # feature_extractor = FeatureExtractor().to(device)

    logger = TensorBoardLogger('log', name='baseline', default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="log/baseline",
        filename="baseline-{epoch:002d}",
        save_top_k=3,
        mode="min",
    )

    model = LitTrainer_conventional(netG=generator, lr=0.01)
    trainer = pl.Trainer(gpus=4, max_epochs=1000, logger=logger, log_every_n_steps=100, strategy='ddp', precision=16,
                         callbacks=[lr_monitor, checkpoint_callback])
    trainer.fit(model, train_dataloaders=training_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()

