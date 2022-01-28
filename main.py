import os
import torch
import numpy as np
import torchio as tio
from trainer_org import LitTrainer
from generator import GeneratorRRDB
from feature_extractor import FeatureExtractor
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

# print(os.getcwd())
# torch.cuda.empty_cache()

def main():
    pl.seed_everything(21011998)

    parser = ArgumentParser()
    parser.add_argument('--std', default=0.3548, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--root_dir', default='/mnt/beta/djboonstoppel/Code', type=str)
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--patch_size', required=True, type=int)
    parser.add_argument('--batch_size', required=True, type=int)

    #precision, gpus, log_every_n_steps, max_epochs, max_time

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitTrainer.add_model_specific_args(parser)
    args = parser.parse_args()

    ### Single config ###
    config = {
        'learning_rate': 1e-4,
        'patch_size': args.patch_size,
        'batch_size': args.batch_size,
        'patients_frac': 0.5,
        'patch_overlap': 0.5,
        'optimizer': 'adam',
        'edge_loss': 2,
        'b1': 0.9,
        'b2': 0.5,
    }

    generator = GeneratorRRDB(channels=1, filters=64, num_res_blocks=1)
    feature_extractor = FeatureExtractor()

    os.makedirs(os.path.join(args.root_dir, 'log', args.name), exist_ok=True)
    logger = TensorBoardLogger('log', name=args.name, default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(args.root_dir, 'log', args.name),
        filename=args.name+"-checkpoint",
        save_top_k=1,
        mode="min",
    )

    model = LitTrainer(netG=generator, netF=feature_extractor, args=args, config=config)

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        max_time=args.max_time,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        # strategy='ddp_spawn',
        precision=args.precision,
        callbacks=[lr_monitor, checkpoint_callback],
        enable_progress_bar=True,
    )

    trainer.fit(
        model,
    )


if __name__ == '__main__':
    main()

