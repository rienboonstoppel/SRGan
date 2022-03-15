import os
from trainer_org_running import LitTrainer
from models.generator import GeneratorRRDB
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
# from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
from ray.tune import CLIReporter
from ray_lightning import RayPlugin
# from ray_lightning.tune import get_tune_ddp_resources
from ray_lightning.tune import TuneReportCallback, get_tune_ddp_resources

# print(os.getcwd())
# torch.cuda.empty_cache()


def train_tune(config, args):
    plugin = RayPlugin(num_workers=4, num_cpus_per_worker=4, use_gpu=False)

    generator = GeneratorRRDB(channels=1, filters=64, num_res_blocks=1)
    # discriminator = Discriminator(input_shape=(1, 64, 64))
    # feature_extractor = FeatureExtractor().to(device)

    log_path = tune.get_trial_dir()

    logger = TensorBoardLogger(save_dir=log_path, name="", version=".")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(log_path, 'checkpoints'),
        filename=args.name+"-checkpoint-{epoch:002d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )

    model = LitTrainer(netG=generator, args=args, config=config)

    trainer = pl.Trainer(
        # gpus=1,
        max_epochs=args.max_epochs,
        max_time=args.max_time,
        logger=logger,
        log_every_n_steps=100,
        # strategy='ddp_spawn',
        precision=args.precision,
        enable_progress_bar=False,
        callbacks=[
            lr_monitor,
            checkpoint_callback,
            TuneReportCallback(
                metrics={'loss': 'val_loss'},
                on="validation_end"),
        ],
        plugins=[plugin]
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
        'learning_rate': tune.grid_search([1e-2, 1e-10]),
        'patch_size': 224,
        'batch_size': 16,
        'patients_frac': 0.5,
        'patch_overlap': 0.5,
    }
    reporter = CLIReporter(
        parameter_columns=["learning_rate", "patch_size", 'batch_size', 'patients_frac', 'patch_overlap'],
        metric_columns=["loss", "training_iteration"])

    # resources_per_trial = {'cpu': 8, 'gpu': 1}
    resources_per_trial = get_tune_ddp_resources(num_workers=4)

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

