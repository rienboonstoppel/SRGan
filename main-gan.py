import os
from trainer_gan import LitTrainer
from models.generator import GeneratorRRDB as GeneratorRRDB
from models.discriminator import Discriminator
from models.feature_extractor import FeatureExtractor
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import timedelta
from utils import print_config


def main():
    pl.seed_everything(21011998)

    parser = ArgumentParser()
    parser.add_argument('--std', default=0.3548, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--root_dir', default='/mnt/beta/djboonstoppel/Code', type=str)
    parser.add_argument('--warmup_batches', default=500, type=int)
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--patch_size', required=True, type=int)
    # parser.add_argument('--gan', action='store_false')

    # --precision=16 --gpus=1 --log_every_n_steps=50 --max_epochs=-1 --max_time="00:00:00:00"

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitTrainer.add_model_specific_args(parser)
    args = parser.parse_args()

    ### Single config ###
    config = {
        'optimizer': 'adam',
        'b1': 0.9,
        'b2': 0.5,
        'batch_size': 16,
        'num_filters': 64,
        'learning_rate': 1e-4,
        'patch_size': args.patch_size,
        'alpha_content': 0,
        'alpha_adversarial': 0.1,
        'ragan': False,
        'gan_mode': 'vanilla',
        'edge_loss': 2,
        'netD_freq': 1,
        'datasource': '1mm_07mm',
        'patients_frac': 0.5,
        'patch_overlap': 0.5,
    }

    print_config(config, args)

    generator = GeneratorRRDB(channels=1, filters=config['num_filters'], num_res_blocks=1)
    discriminator = Discriminator(input_shape=(1, config['patch_size'], config['patch_size']))
    feature_extractor = FeatureExtractor()

    os.makedirs(os.path.join(args.root_dir, 'log', args.name), exist_ok=True)
    logger = TensorBoardLogger('log', name=args.name, default_hp_metric=False)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback_best = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(args.root_dir, 'log', args.name),
        filename=args.name+"-checkpoint-best",
        save_top_k=1,
        mode="min",
    )

    checkpoint_callback_time = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename=args.name+"-checkpoint-{epoch}",
        save_top_k=-1,
        train_time_interval=timedelta(hours=3),
    )

    model = LitTrainer(netG=generator, netF=feature_extractor, netD=discriminator, args=args, config=config)

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        max_time=args.max_time,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        # strategy=DDPPlugin(find_unused_parameters=True),
        precision=args.precision,
        callbacks=[lr_monitor, checkpoint_callback_best, checkpoint_callback_time],
        enable_progress_bar=True,
    )

    trainer.fit(
        model,
    )

if __name__ == '__main__':
    main()

