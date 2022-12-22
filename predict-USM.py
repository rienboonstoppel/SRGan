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
from models.generator_DeepUResnet_v2 import build_deepunet as generator_DeepUResnet_v2
from models.discriminator import Discriminator
from models.feature_extractor import FeatureExtractor
from argparse import ArgumentParser
from utils import save_to_nifti, save_to_nifti_pp
from dataset_tio import sim_data, MRBrainS18_data, HCP_data, OASIS_data
from transform import Normalize
from tqdm import tqdm
import json

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

project = 'example'
exp_name = 'mixed-wgan'
ckpt_path = os.path.join('log', project, exp_name, exp_name+'-checkpoint-best.ckpt')

output_folder = os.path.join('output', project, exp_name)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def main(ckpt_path, output_folder):
    parser = ArgumentParser()
    parser.add_argument('--root_dir', default='/mnt/beta/djboonstoppel/Code', type=str)
    parser.add_argument('--gan', action='store_true')
    parser.add_argument('--source', required=True, type=str, choices=['sim', 'mrbrains', 'hcp', 'oasis'])
    parser.add_argument('--generator', default='ESRGAN', type=str,
                        choices=['ESRGAN', 'RRDB', 'DeepUResnet', 'DeepUResnet_v2', 'FSRCNN'])
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
    elif args.generator == 'DeepUResnet_v2':
        generator = generator_DeepUResnet_v2()
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
                                           every_other=args.every_other,
                                           augment=False)
    elif args.source == 'hcp':
        subjects, subjects_info = HCP_data(dataset=dataset,
                                           root_dir=data_path,
                                           middle_slices=args.middle_slices,
                                           every_other=args.every_other,
                                           augment=False)
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

    path = os.path.join(args.root_dir, ckpt_path)
    if args.gan:
        model = LitTrainer_gan.load_from_checkpoint(
            netG=generator,
            netF=feature_extractor,
            netD=discriminator,
            checkpoint_path=path,
        )
        if model.hparams.config['gan_mode'] == 'wgan':
            mode = 'wgan'
        elif model.hparams.config['gan_mode'] == 'vanilla':
            mode = 'rasgan'

    else:
        model = LitTrainer_org.load_from_checkpoint(
            netG=generator,
            netF=feature_extractor,
            checkpoint_path=path,
        )

    print('Predicting images using model checkpoint trained with config:')
    print_args = ['nr_hcp_train', 'nr_sim_train', 'alpha_pixel', 'alpha_edge', 'alpha_perceptual',
                  'alpha_adversarial', 'alpha_gradientpenalty', 'generator', 'num_res_blocks']
    print("{:<22}| {:<10}".format('Var', 'Value'))
    print('-' * 32)
    for arg in print_args:
        print("{:<22}| {:<10} ".format(arg, model.hparams.config[arg]))
    if args.gan:
        print("{:<22}| {:<10} ".format('gan_mode', mode))

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

        folder_name = 'sim={}_hcp={}_mode={}'.format(
            model.nr_sim_train,
            model.nr_hcp_train,
            mode)

        output_path = os.path.join(output_folder,
                                   folder_name,
                                   args.source,
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

        mask = subjects[i]['MSK'][tio.DATA].numpy()[0]
        bg_idx = np.where(mask == 0)

        sr = sr['SR'][tio.DATA].numpy()[0]
        sr[bg_idx] = 0

        save_to_nifti(img=sr,
                      header=subjects_info[i]['LR']['header'],
                      max_val=subjects_info[i]['LR']['scaling'],
                      fname=os.path.join(output_path,
                                         img_fname + '_SR.nii.gz'),
                      source=args.source,
                      )

        # save_to_nifti_pp(img=sr,
        #                  header=subjects_info[i]['LR']['header'],
        #                  max_val=subjects_info[i]['LR']['scaling'],
        #                  fname=os.path.join(output_path,
        #                                     img_fname + '_SR_USM05.nii.gz'),
        #                  source=args.source,
        #                  )

        lr = subjects[i]['LR'][tio.DATA].numpy()[0]
        lr[bg_idx] = 0

        save_to_nifti(img=lr,
                      header=subjects_info[i]['LR']['header'],
                      max_val=subjects_info[i]['LR']['scaling'],
                      fname=os.path.join(output_path,
                                         img_fname + '_LR.nii.gz'),
                      source=args.source,
                      )


if __name__ == '__main__':
    main(ckpt_path, output_folder)
