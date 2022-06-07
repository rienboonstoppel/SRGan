import numpy as np

from trainer_tune_test import LitTrainer
from models.generator_ESRGAN import GeneratorRRDB
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# print(os.getcwd())
# torch.cuda.empty_cache()


def train_tune(config, args):
    pl.seed_everything(21011998)

    generator = GeneratorRRDB(channels=1, filters=64, num_res_blocks=1)
    # discriminator = Discriminator(input_shape=(1, 64, 64))
    # feature_extractor = FeatureExtractor().to(device)

    logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='/mnt/beta/djboonstoppel/Code/log/baseline',
        filename='baseline-{epoch:002d}',
        save_top_k=3,
        mode='min',
    )

    metrics = {'loss': 'val_loss'}

    model = LitTrainer(netG=generator, config=config, args=args)

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        max_time=args.max_time,
        logger=logger,
        log_every_n_steps=100,
        # strategy='ddp_spawn',
        precision=args.precision,
        callbacks=[
            TuneReportCallback(metrics=metrics, on="validation_end"),
        ],
        enable_progress_bar=False,
    )

    trainer.fit(
        model,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--patients_frac', default=1.0, type=float)
    parser.add_argument('--std', default=0.3548, type=float)
    parser.add_argument('--patch_size', default=64, type=int)
    parser.add_argument('--patch_overlap', default=.5, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--root_dir', default='/mnt/beta/djboonstoppel/Code/data', type=str)
    parser.add_argument('--num_samples', required=True, type=int)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitTrainer.add_model_specific_args(parser)
    args = parser.parse_args()

    patients_fracs = np.around(np.arange(0.1, 0.6, 0.1), 1)
    patch_size = [32, 64, 128]
    batch_size = [1024, 256, 64]

    def _iter():
        for patients_frac in patients_fracs:
            for i in range(len(patch_size)):
                yield patients_frac, patch_size[i], batch_size[i]

    config = {
        "learning_rate": 1e-2,
        "frac_patch_batch": tune.grid_search(list(_iter())),
    }

    scheduler = ASHAScheduler(
        max_t=args.max_epochs,
        grace_period=50,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["learning_rate", "frac_patch_batch"],
        metric_columns=["loss", "training_iteration"])

    resources_per_trial = {'cpu': 8, 'gpu': args.gpus}

    train_fn_with_parameters = tune.with_parameters(
        train_tune,
        args=args
        )

    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=config,
        num_samples=args.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name='tune_patients_fracs2',
        verbose=1,
        local_dir='/mnt/beta/djboonstoppel/Code/ray_results',
    )

    print('Best hyperparameters found were: ', analysis.best_config)


if __name__ == '__main__':
    main()

