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
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
from ray.tune import CLIReporter
from datetime import timedelta
from pytorch_lightning.plugins import DDPPlugin
import warnings
import json

warnings.filterwarnings('ignore', '.*The dataloader, .*')

# print(os.getcwd())
# torch.cuda.empty_cache()


def train_tune(config, args):
    generator = GeneratorRRDB(channels=1, filters=64, num_res_blocks=1)
    feature_extractor = FeatureExtractor()

    log_path = tune.get_trial_dir()
    logger = TensorBoardLogger(save_dir=log_path, name="", version=".")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    ckpt_path = os.path.join(args.root_dir, 'ray_results', args.name, 'checkpoints')
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_filename = 'checkpoint_{}'.format(config['alpha_content'],
                                           )

    # ckpt_filename = 'checkpoint_{}_{}_{}_{}_{}_{}'.format(config['patch_size'],
    #                                                       config['batch_size'],
    #                                                       config['patients_frac'],
    #                                                       config['edge_loss'],
    #                                                       config['optimizer'],
    #                                                       config['learning_rate'],
    #                                                       config['b1'],
    #                                                       config['b2'],
    #                                                       config['alpha_content']
    #                                                       )

    checkpoint_callback_best = ModelCheckpoint(
        monitor="val_loss",
        dirpath=ckpt_path,
        filename=ckpt_filename + "-best",
        save_top_k=1,
        mode="min",
    )

    checkpoint_callback_time = ModelCheckpoint(
        dirpath=ckpt_path,
        filename=ckpt_filename + "-{epoch}",
        save_top_k=-1,
        train_time_interval=timedelta(hours=2),
    )


    model = LitTrainer(netG=generator, netF=feature_extractor, args=args, config=config)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.max_epochs,
        max_time=args.max_time,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        # strategy=DDPPlugin(find_unused_parameters=False),
        precision=args.precision,
        enable_progress_bar=False,
        callbacks=[
            lr_monitor,
            checkpoint_callback_best,
            checkpoint_callback_time,
            TuneReportCallback(
                metrics={'loss': 'val_loss'},
                on="validation_end"),
        ],
    )

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
    parser.add_argument('--patch_size', required=True, type=int)
    parser.add_argument('--batch_size', required=True, type=int)

    # precision, log_every_n_steps, max_epochs, max_time

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitTrainer.add_model_specific_args(parser)
    args = parser.parse_args()

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
        'alpha_content': tune.grid_search([0, 0.01, 0.1, 0.5, 1, 5]),
    }

    reporter = CLIReporter(
        parameter_columns=['alpha_content'],
        metric_columns=["loss", "training_iteration"])

    resources_per_trial = {'cpu': 8, 'gpu': 1}

    train_fn_with_parameters = tune.with_parameters(
        train_tune,
        args=args,
    )

    def trial_dirname_string(trial):
        fname = json.dumps(trial.evaluated_params)
        chars = ['"', '{', '}', '. ', '_']
        for char in chars:
            fname = fname.replace(char, '')
        fname = fname.replace(': ', '_')
        fname = fname.replace(', ', '-')
        fname = fname.replace('.', ',')
        return fname

    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        config=config,
        num_samples=args.num_samples,
        progress_reporter=reporter,
        name=args.name,
        trial_dirname_creator=trial_dirname_string,
        local_dir=os.path.join(args.root_dir, 'ray_results', ),
        verbose=1,
        metric="loss",
        mode="min",

    )

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == '__main__':
    main()
