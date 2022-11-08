import os
import numpy as np
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
from utils import save_to_nifti
from dataset_tio import sim_data, MRBrainS18_data, HCP_data, OASIS_data
from transform import Normalize
from glob import glob
from tqdm import tqdm
import json

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Sweep
# run_id = 62
# ckpt_path = glob('log/sweep-2/*/*'+str(run_id)+'*')[0]

# run_ids = np.arange(13,15)
run_ids = [13]
ckpt_paths = [glob('log/final-checks/*/*-*-' + str(run_id) + '-checkpoint-best.ckpt')[0] for run_id in run_ids]
ckpt_paths.append('log/final-checks/winter-morning-15/winter-morning-15-checkpoint-epoch=7.ckpt')

output_folder = 'output/exp2a-gans'


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def main(ckpt_paths, output_folder):
    parser = ArgumentParser()
    parser.add_argument('--root_dir', default='/mnt/beta/djboonstoppel/Code', type=str)
    parser.add_argument('--gan', action='store_true')
    parser.add_argument('--source', required=True, type=str, choices=['sim', 'mrbrains', 'hcp', 'oasis'])
    parser.add_argument('--generator', default='ESRGAN', type=str, choices=['ESRGAN', 'RRDB', 'DeepUResnet', 'FSRCNN'])
    parser.add_argument('--num_filters', default=64, type=int)
    parser.add_argument('--patch_size', default=64, type=int)
    parser.add_argument('--patch_overlap', default=0.1, type=float)
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

    dataset = 'test'

    if args.source == 'sim':
        subjects, subjects_info = sim_data(dataset=dataset,
                                           root_dir=data_path,
                                           middle_slices=args.middle_slices,
                                           every_other=args.every_other)
    elif args.source == 'hcp':
        subjects, subjects_info = HCP_data(dataset=dataset,
                                           root_dir=data_path,
                                           middle_slices=args.middle_slices,
                                           every_other=args.every_other)
    elif args.source == 'oasis':
        subjects, subjects_info = OASIS_data(dataset=dataset,
                                             root_dir=data_path,
                                             middle_slices=args.middle_slices,
                                             every_other=args.every_other,
                                             augment=False)
    elif args.source == 'mrbrains':
        subjects, subjects_info = MRBrainS18_data(dataset=dataset,
                                                  root_dir=data_path,
                                                  middle_slices=args.middle_slices,
                                                  every_other=args.every_other,
                                                  augment=False)
    else:
        raise ValueError("Dataset '{}' not implemented".format(args.source))

    transform = tio.Compose([
        Normalize(std=args.std),
    ])
    predict_set = tio.SubjectsDataset(
        subjects, transform=transform)

    grid_samplers = []
    for i in range(len(subjects)):
        grid_sampler = tio.inference.GridSampler(
            predict_set[i],
            patch_size=(subjects[i]['LR'][tio.DATA].shape[1], subjects[i]['LR'][tio.DATA].shape[2], 1),
            patch_overlap=0,
            padding_mode=0,
        )
        grid_samplers.append(grid_sampler)

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

        print('Predicting images using model checkpoint trained with config:')
        print_args = ['nr_hcp_train', 'nr_sim_train', 'alpha_pixel', 'alpha_edge', 'alpha_perceptual',
                      'alpha_adversarial', 'alpha_gradientpenalty', 'generator', 'num_res_blocks', 'gan_mode', 'ragan']
        print("{:<22}| {:<10}".format('Var', 'Value'))
        print('-' * 32)
        for arg in print_args:
            print("{:<22}| {:<10} ".format(arg, model.hparams.config[arg]))

        model.to(device)
        model.eval()

        for i in tqdm(range(len(grid_samplers))):
            if args.source == 'sim':
                img_fname = "08-Apr-2022_Ernst_labels_{:06d}_3T_T1w_MPR1_img_act_1_contrast_1".format(
                    subjects_info[i]['id'])
            elif args.source == 'hcp':
                img_fname = "{:06d}_3T_T1w_MPR1_img".format(subjects_info[i]['id'])
            elif args.source == 'oasis':
                img_fname = "OAS1_{:04d}_MR1_mpr_n4_anon_111_t88_masked_gfc".format(subjects_info[i]['id'])
            elif args.source == 'mrbrains':
                img_fname = "p{:01d}_reg_T1".format(subjects_info[i]['id'])
            else:
                raise ValueError("Dataset '{}' not implemented".format(args.source))

            folder_name = 'sim={}_hcp={}_mode={}_ragan={}'.format(
                model.nr_sim_train,
                model.nr_hcp_train,
                model.hparams.config['gan_mode'],
                model.hparams.config['ragan'],
            )

            # folder_name = 'px={}_edge={}_vgg={}_gan={}_mode={}_ragan={}' \
            # folder_name = 'gen={}_blocks={}_mode={}_ragan={}' \
            #             .format(
                # model.hparams.config['alpha_pixel'],
                # model.hparams.config['alpha_edge'],
                # model.hparams.config['alpha_perceptual'],
                # model.hparams.config['alpha_adversarial'],
                # args.generator,
                # model.hparams.config['num_res_blocks'],
                # model.hparams.config['gan_mode'],
                # model.hparams.config['ragan'],
                #         ).replace('.', '')


            # name = 'mode={}_ragan={}_blocks={}_updated'.format(
            #     model.hparams.config['gan_mode'],
            #     model.hparams.config['ragan'],
            #     model.hparams.config['num_res_blocks'])

            output_path = os.path.join(output_folder,
                                       args.source,
                                       folder_name,
                                       dataset)
            os.makedirs(output_path, exist_ok=True)

            with open(os.path.join(output_path, 'config.json'), 'w') as outfile:
                json.dump(dict(model.hparams.config), outfile, indent=4)

            aggregator = tio.inference.GridAggregator(grid_samplers[i])

            patch_loader = torch.utils.data.DataLoader(
                grid_samplers[i], batch_size=model.hparams.config['batch_size'])

            with torch.no_grad():
                for patches_batch in patch_loader:
                    imgs_lr = patches_batch['LR'][tio.DATA].squeeze(4)
                    imgs_sr = model(imgs_lr.to(device)).unsqueeze(4)

                    locations = patches_batch[tio.LOCATION]
                    aggregator.add_batch(imgs_sr, locations)

            foreground = aggregator.get_output_tensor() * args.std
            generated = tio.ScalarImage(tensor=foreground)
            sr = tio.Subject({'SR': generated})

            save_to_nifti(img=sr['SR'],
                          header=subjects_info[i]['LR']['header'],
                          max_val=subjects_info[i]['LR']['scaling'],
                          fname=os.path.join(output_path,
                                             img_fname + '_SR.nii.gz'),
                          source=args.source,
                          )


if __name__ == '__main__':
    main(ckpt_paths, output_folder)
