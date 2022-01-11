import os
import torch
import numpy as np
import torchio as tio
from trainer_org_running import LitTrainer
from generator import GeneratorRRDB
from discriminator import Discriminator
from feature_extractor import FeatureExtractor
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray import tune
from ray.tune import CLIReporter
from datetime import timedelta
from ray.rllib import _register_all
_register_all()

# print(os.getcwd())
# torch.cuda.empty_cache()


def train_tune(config, args, checkpoint_dir=None):
    generator = GeneratorRRDB(channels=1, filters=64, num_res_blocks=1)
    discriminator = Discriminator(input_shape=(1, config['patch_size'], config['patch_size']))
    feature_extractor = FeatureExtractor()

    log_path = tune.get_trial_dir()
    logger = TensorBoardLogger(save_dir=log_path, name="", version=".")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    ckpt_path = os.path.join(args.root_dir, 'ray_results', args.name, 'checkpoints')
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_filename = 'checkpoint_{}_{}_{}'.format(config['patch_size'], config['batch_size'], config['patients_frac'])

    checkpoint_callback_best = ModelCheckpoint(
        monitor="val_loss",
        dirpath=ckpt_path,
        filename=ckpt_filename+"-best",
        save_top_k=1,
        mode="min",
    )

    checkpoint_callback_time = ModelCheckpoint(
        dirpath=os.path.join(log_path, 'checkpoints'),
        filename=ckpt_filename+"-{epoch:002d}",
        train_time_interval=timedelta(hours=1),
    )

    kwargs = {
        'gpus': 1,
        'max_epochs': args.max_epochs,
        'max_time': args.max_time,
        'logger': logger,
        'log_every_n_steps': 100,
        'precision': args.precision,
        'enable_progress_bar': False,
        'callbacks': [
            lr_monitor,
            checkpoint_callback_best,
            checkpoint_callback_time,
            TuneReportCallback(
                metrics={'loss': 'val_loss'},
                on="validation_end"),
            TuneReportCheckpointCallback(
                metrics={
                    "loss": "val_loss",
                },
                filename="checkpoint",
                on="validation_end")
        ],
    }

    if checkpoint_dir:
        kwargs["resume_from_checkpoint"] = os.path.join(
            checkpoint_dir, "checkpoint")

    model = LitTrainer(netG=generator, netF=feature_extractor, netD=discriminator, args=args, config=config)

    trainer = pl.Trainer(**kwargs)

    trainer.fit(
        model,
    )



def main():
    pl.seed_everything(21011998)

    parser = ArgumentParser()
    parser.add_argument('--std', default=0.3548, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--root_dir', default='/mnt/beta/djboonstoppel/Code', type=str)
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--num_samples', required=True, type=int)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitTrainer.add_model_specific_args(parser)
    args = parser.parse_args()

    # ### Single config ###
    # config = {
    #     'learning_rate': 0.01,
    #     'patch_size': 64,
    #     'batch_size': 256,
    #     'patients_frac': 0.5,
    #     'patch_overlap': 0.5,
    # }

    ### Gridsearch ###
    config = {
        'learning_rate': 1e-2,
        'patch_size': tune.grid_search([64, 224]),
        'batch_size': tune.sample_from(lambda spec: 256 if (spec.config.patch_size == 64) else 16),
        'patients_frac': tune.grid_search([0.1, 0.5, 1.0]),
        'patch_overlap': 0.5,
    }
    reporter = CLIReporter(
        parameter_columns=["learning_rate", "patch_size", 'batch_size', 'patients_frac', 'patch_overlap'],
        metric_columns=["loss", "training_iteration"])

    resources_per_trial = {'cpu': 8, 'gpu': 1}

    train_fn_with_parameters = tune.with_parameters(
        train_tune,
        args=args
        )

    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        config=config,
        num_samples=args.num_samples,
        progress_reporter=reporter,
        name=args.name,
        local_dir=os.path.join(args.root_dir, 'ray_results',),
        verbose=1,
        metric="loss",
        mode="min",

    )

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == '__main__':
    main()

