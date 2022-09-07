import os
import torch
import torchio as tio
from trainer_org import LitTrainer as LitTrainer_org
from trainer_gan import LitTrainer as LitTrainer_gan
from models.generator_ESRGAN import GeneratorRRDB as generator_ESRGAN
from models.generator_FSRCNN import FSRCNN as generator_FSRCNN
from models.generator_RRDB import GeneratorRRDB as generator_RRDB
from models.generator_DeepUResnet import DeepUResnet as generator_DeepUResnet
from models.discriminator import Discriminator
from models.feature_extractor import FeatureExtractor
from argparse import ArgumentParser
from utils import print_config, save_subject, save_to_nifti
from dataset_tio import SimImage, MRBrainS18Image, HCPImage, OASISImage, calculate_overlap, sim_data, \
    MRBrainS18_data, HCP_data, OASIS_data
from transform import Normalize, RandomIntensity, RandomGamma, RandomBiasField
from metrics import get_scores
import time
from glob import glob
from tqdm import tqdm

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# ## Single runs
# exp_name = 'test-checkpoint-1'
# epoch = 4
# ckpt_fname1 = '{}-checkpoint-epoch={}.ckpt'.format(exp_name, 1)
# ckpt_fname2 = '{}-checkpoint-epoch={}.ckpt'.format(exp_name, 4)
# ckpt_paths = [os.path.join('log', exp_name, ckpt_fname1), os.path.join('log', exp_name, ckpt_fname2)]

# Sweep
# run_id = 62
# ckpt_path = glob('log/sweep-2/*/*'+str(run_id)+'*')[0]

run_ids = [1,2,3,4]
ckpt_paths = [glob('log/sweep-2/*/*'+str(run_id)+'*')[0] for run_id in run_ids]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def main(ckpt_paths):
    parser = ArgumentParser()
    parser.add_argument('--root_dir', default='/mnt/beta/djboonstoppel/Code', type=str)
    parser.add_argument('--gan', action='store_true')
    parser.add_argument('--source', required=True, type=str, choices=['sim', 'mrbrains', 'hcp', 'oasis'])
    parser.add_argument('--generator', default='ESRGAN', type=str, choices=['ESRGAN', 'RRDB', 'DeepUResnet', 'FSRCNN'])
    parser.add_argument('--num_filters', default=64, type=int)
    parser.add_argument('--patch_size', default=64, type=int)
    parser.add_argument('--patch_overlap', default=0.5, type=float)
    parser.set_defaults(gan=False)

    args = parser.parse_args()

    if args.gan:
        parser = LitTrainer_gan.add_model_specific_args(parser)
        args = parser.parse_args()
    else:
        parser = LitTrainer_org.add_model_specific_args(parser)
        args = parser.parse_args()

    if args.generator == 'ESRGAN':
        generator = generator_ESRGAN(channels=1, filters=args.num_filters, num_res_blocks=1)
    elif args.generator == 'RRDB':
        generator = generator_RRDB(channels=1, filters=args.num_filters, num_res_blocks=1)
    elif args.generator == 'DeepUResnet':
        generator = generator_DeepUResnet(nrfilters=args.num_filters)
    elif args.generator == 'FSRCNN':
        generator = generator_FSRCNN(scale_factor=1)
    else:
        raise NotImplementedError(
            "Generator architecture '{}' is not recognized or implemented".format(args.generator))

    discriminator = Discriminator(input_shape=(1, args.patch_size, args.patch_size))
    feature_extractor = FeatureExtractor()

    data_path = os.path.join(args.root_dir, 'data')

    args.middle_slices = None

    if args.source == 'sim':
        val_subjects, subjects_info = sim_data(dataset='validation',
                                root_dir=data_path,
                                middle_slices=args.middle_slices,
                                every_other=args.every_other)
    elif args.source == 'hcp':
        val_subjects, subjects_info = HCP_data(dataset='validation',
                                root_dir=data_path,
                                middle_slices=args.middle_slices,
                                every_other=args.every_other)
    elif args.source == 'oasis':
        val_subjects, subjects_info = OASIS_data(dataset='validation',
                                  root_dir=data_path,
                                  middle_slices=args.middle_slices,
                                  every_other=args.every_other)
    elif args.source == 'mrbrains':
        val_subjects, subjects_info = MRBrainS18_data(dataset='validation',
                                       root_dir=data_path,
                                       middle_slices=args.middle_slices,
                                       every_other=args.every_other)
    else:
        raise ValueError("Dataset '{}' not implemented".format(args.source))

    overlap, nr_patches = calculate_overlap(val_subjects[0],
                                            (args.patch_size, args.patch_size),
                                            (args.patch_overlap, args.patch_overlap))
    val_transform = tio.Compose([
        Normalize(std=args.std),
    ])
    val_set = tio.SubjectsDataset(
        val_subjects, transform=val_transform)

    grid_samplers = []
    for i in range(len(val_subjects)):
        grid_sampler = tio.inference.GridSampler(
            val_set[i],
            patch_size=(args.patch_size, args.patch_size, 1),
            patch_overlap=overlap,
            padding_mode=0,
        )
        grid_samplers.append(grid_sampler)

    # j = 1
    for ckpt_path in ckpt_paths:
        path = os.path.join(args.root_dir, ckpt_path)
        if args.gan:
            model = LitTrainer_gan.load_from_checkpoint(
                netG=generator,
                netF=feature_extractor,
                netD=discriminator,
                checkpoint_path=path,
            )
        else:
            model = LitTrainer_org.load_from_checkpoint(
                netG=generator,
                netF=feature_extractor,
                checkpoint_path=path,
            )
        print('Checkpoint trained on {} hcp subjects and {} sim subjects'.format(model.hparams.config['nr_hcp_train'],
                                                                                 model.hparams.config['nr_sim_train']))

        model.to(device)
        model.eval()

        for i in tqdm(range(len(grid_samplers))):
            output_path = os.path.join('output/sweep-2', args.source + '_' + str(subjects_info[i]['id']))
            os.makedirs(output_path, exist_ok=True)

            aggregator = tio.inference.GridAggregator(grid_samplers[i])  # , overlap_mode='average')

            patch_loader = torch.utils.data.DataLoader(
                grid_samplers[i], batch_size=model.hparams.config['batch_size'])

            # start_time = time.time()

            with torch.no_grad():
                for patches_batch in patch_loader:
                    imgs_lr = patches_batch['LR'][tio.DATA].squeeze(4)
                    imgs_sr = model(imgs_lr.to(device)).unsqueeze(4)

                    locations = patches_batch[tio.LOCATION]
                    aggregator.add_batch(imgs_sr, locations)

            # end_time = time.time()
            # print('Time: {:.10f} s'.format(end_time - start_time))

            foreground = aggregator.get_output_tensor() * args.std
            generated = tio.ScalarImage(tensor=foreground)
            sr = tio.Subject({'SR': generated})

            save_to_nifti(img=sr['SR'],
                          header=subjects_info[i]['LR']['header'],
                          max_val=subjects_info[i]['LR']['scaling'],
                          fname=os.path.join(output_path, 'SR_hcp{}_sim{}.nii.gz'.format(model.hparams.config['nr_hcp_train'],
                                                                                         model.hparams.config['nr_sim_train'])),
                          source=args.source,
                          )

    # save_to_nifti(img=val_subjects[0]['LR'],
    #               header=subjects_info[0]['LR']['header'],
    #               max_val=subjects_info[0]['LR']['scaling'],
    #               fname=os.path.join(output_path, 'LR.nii.gz'),
    #               source=args.source,
    #               )
    #
    # if args.source == 'sim' or args.source == 'hcp':
    #     save_to_nifti(img=val_subjects[0]['HR'],
    #                   header=subjects_info[0]['HR']['header'],
    #                   max_val=subjects_info[0]['HR']['scaling'],
    #                   fname=os.path.join(output_path, 'HR.nii.gz'),
    #                   source=args.source,
    #                   )


if __name__ == '__main__':
    main(ckpt_paths)
