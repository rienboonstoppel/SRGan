#%%

import torch
# import numpy as np
import torchio as tio
from archive.trainer_gan import LitTrainer
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset_tio import data_split, Normalize, calculate_overlap

train_subjects = data_split('training')
val_subjects = data_split('validation')

std = 0.3548

training_transform = tio.Compose([
    Normalize(std=std),
    # tio.RandomNoise(p=0.5),
    # tio.RandomFlip(),
])

training_set = tio.SubjectsDataset(
    train_subjects, transform=training_transform)

batch_size = 32
training_batch_size = batch_size
validation_batch_size = batch_size

num_workers = 10
patch_size = (64,64)
ovl_perc = (.5, .5)
overlap, nr_patches = calculate_overlap(train_subjects[0]['LR'], patch_size, ovl_perc)
samples_per_volume = nr_patches

max_queue_length = samples_per_volume*10
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

training_loader = torch.utils.data.DataLoader(
    training_queue, batch_size=training_batch_size)

model = LitTrainer()
model = model.cuda()
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_dataloader=training_loader)