import os
from trainer_gan import LitTrainer
from models.generator import GeneratorRRDB
from models.discriminator import Discriminator
from models.feature_extractor import FeatureExtractor
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
from ray.tune import CLIReporter
from datetime import timedelta
import warnings
import json

warnings.filterwarnings('ignore', '.*The dataloader, .*')


# print(os.getcwd())
# torch.cuda.empty_cache()


def train_tune(config, args):
    generator = GeneratorRRDB(channels=1, filters=config['num_filters'], num_res_blocks=config['num_res_blocks'])
    discriminator = Discriminator(input_shape=(1, config['patch_size'], config['patch_size']))
    feature_extractor = FeatureExtractor()

    log_path = tune.get_trial_dir()
    logger = TensorBoardLogger(save_dir=log_path, name="", version=".")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    ckpt_path = os.path.join(args.root_dir, 'ray_results', args.name, 'checkpoints')
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_filename = 'checkpoint_{}'.format(config['num_real'])

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
        # train_time_interval=timedelta(hours=3),
        every_n_epochs=3,
    )

    model = LitTrainer(netG=generator, netF=feature_extractor, netD=discriminator, args=args, config=config)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.max_epochs,
        max_time=args.max_time,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        # strategy=DDPPlugin(find_unused_parameters=True),
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
    parser.add_argument('--warmup_batches', default=1000, type=int)

    # precision, log_every_n_steps, max_epochs, max_time

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitTrainer.add_model_specific_args(parser)
    args = parser.parse_args()

    config = {
        'optimizer': 'adam',
        'b1': 0.9,
        'b2': 0.5,
        'batch_size': 16,
        'num_filters': 64,
        'learning_rate_G': 5e-5,
        'learning_rate_D': 5e-5,
        'patch_size': args.patch_size,
        'alpha_content': 1,
        'alpha_adversarial': 0.1,
        'ragan': False,
        'gan_mode': 'vanilla',
        'edge_loss': 2,
        'netD_freq': 1,
        'datasource': '2mm_1mm',
        'patients_frac': .3,
        'patch_overlap': 0.5,
        'num_patients': 30,
        'num_real': tune.grid_search([0,1,2,3,4,5,6]),
        'num_res_blocks': 1,
    }
    reporter = CLIReporter(
        parameter_columns=['num_real'],
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
