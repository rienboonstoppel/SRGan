import os
from trainer_tune import LitTrainer
from models.generator_ESRGAN import GeneratorRRDB
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback


# print(os.getcwd())
# torch.cuda.empty_cache()


def main():
    parser = ArgumentParser()
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--root_dir', default='/mnt/beta/djboonstoppel/Code', type=str)
    parser.add_argument('--num_samples', required=True, type=int)
    parser.add_argument('--std', type=float, default=-.3548)
    parser.add_argument('--name', required=True, type=str)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitTrainer.add_model_specific_args(parser)
    args = parser.parse_args()

    ### Single config ###
    config = {
        'learning_rate': 0.01,
        'patch_size': 64,
        'batch_size': 256,
        'patients_frac': 0.5,
        'patch_overlap': 0.5,
    }

    # path = tune.get_trial_dir
    path = '/mnt/beta/djboonstoppel/Code/ray_results/patch-test-v3'
    pl.seed_everything(21011998)

    generator = GeneratorRRDB(channels=1, filters=64, num_res_blocks=1)
    # discriminator = Discriminator(input_shape=(1, 64, 64))
    # feature_extractor = FeatureExtractor().to(device)

    logger = TensorBoardLogger(save_dir=path, name="", version=".")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(path, 'checkpoints'),
        filename=args.name+"-checkpoint-{epoch:002d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    metrics = {'loss': 'val_loss'}

    model = LitTrainer(netG=generator, config=config, args=args)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.max_epochs,
        max_time=args.max_time,
        logger=logger,
        log_every_n_steps=100,
        # strategy='ddp_spawn', # THIS BREAKS RAY[TUNE]!!!
        precision=args.precision,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            TuneReportCallback(
                metrics=metrics,
                on="validation_end"),
            TuneReportCheckpointCallback(
                metrics=metrics,
                filename="checkpoint_tune",
                on="validation_end")
        ],
        enable_progress_bar=False,
    )

    trainer.fit(
        model,
    )


# def main():



    # train_tune(config, args)

    # ### Gridsearch ###
    # config = {
    #     'learning_rate': 0.01,
    #     'patch_size': tune.grid_search([64, 224]),
    #     'batch_size': tune.sample_from(lambda spec: 256 if (spec.config.patch_size == 64) else 16),
    #     'patients_frac': 0.5,
    #     'patch_overlap': 0.5,
    # }
    #
    # scheduler = ASHAScheduler(
    #     max_t=args.max_epochs,
    #     grace_period=250,
    #     reduction_factor=2)
    #
    # reporter = CLIReporter(
    #     parameter_columns=["learning_rate", "patch_size", 'batch_size', 'patients_frac', 'patch_overlap'],
    #     metric_columns=["loss", "training_iteration"])
    #
    # resources_per_trial = {'cpu': 8, 'gpu': 1}
    #
    # train_fn_with_parameters = tune.with_parameters(
    #     train_tune,
    #     args=args
    #     )
    #
    # analysis = tune.run(
    #     train_fn_with_parameters,
    #     resources_per_trial=resources_per_trial,
    #     metric="loss",
    #     mode="min",
    #     config=config,
    #     num_samples=args.num_samples,
    #     scheduler=scheduler,
    #     progress_reporter=reporter,
    #     name=args.name,
    #     local_dir=os.path.join(args.root_dir, 'ray_results',),
    #     verbose=1,
    # )
    #
    # print('Best hyperparameters found were: ', analysis.best_config)


if __name__ == '__main__':
    main()

