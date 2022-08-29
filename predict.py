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
from utils import print_config, save_subject
from dataset_tio import SimImage, MRBrainS18Image, HCPImage, OASISImage, calculate_overlap, sim_data, \
    MRBrainS18_data, HCP_data
from transform import Normalize, RandomIntensity, RandomGamma, RandomBiasField
from metrics import get_scores
import time


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

### Single config ###
default_config = {
    'optimizer': 'adam',
    'b1': 0.9,
    'b2': 0.5,
    'batch_size': 16,
    'num_filters': 64,
    'learning_rate_G': 2e-5,
    'learning_rate_D': 2e-5,
    'patch_size': 64,
    'alpha_edge': 0.3,
    'alpha_pixel': 0.7,
    'alpha_perceptual': 1,
    'alpha_adversarial': 0.1,
    'ragan': False,
    'gan_mode': 'vanilla',
    'edge_loss': 2,
    'netD_freq': 1,
    'data_source': 'sim',
    'data_resolution': '1mm_07mm',
    'patients_frac': 0.5,
    'patch_overlap': 0.5,
    'generator': 'ESRGAN'
}

exp_name = 'mixed-30-30_everyother'
epoch = 2
ckpt_fname = '{}-checkpoint-epoch={}.ckpt'.format(exp_name, epoch)
ckpt_path = os.path.join('log', exp_name, ckpt_fname)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def main(config, ckpt_path):
    parser = ArgumentParser()
    parser.add_argument('--root_dir', default='/mnt/beta/djboonstoppel/Code', type=str)
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--gan', action='store_true')
    parser.add_argument('--num', required=True, type=int)
    parser.add_argument('--source', required=True, type=str, choices=['sim', 'mrbrains18', 'hcp', 'hcp_gen', 'oasis'])
    parser.set_defaults(gan=False)

    args = parser.parse_args()

    if args.gan:
        parser = LitTrainer_gan.add_model_specific_args(parser)
        args = parser.parse_args()
    else:
        parser = LitTrainer_org.add_model_specific_args(parser)
        args = parser.parse_args()

    config['patients_dist'] = (30,10,10)
    att_config = AttrDict(config)

    if att_config.generator == 'ESRGAN':
        generator = generator_ESRGAN(channels=1, filters=att_config.num_filters, num_res_blocks=1)
    elif att_config.generator == 'RRDB':
        generator = generator_RRDB(channels=1, filters=att_config.num_filters, num_res_blocks=1)
    elif att_config.generator == 'DeepUResnet':
        generator = generator_DeepUResnet(nrfilters=att_config.num_filters)
    elif att_config.generator == 'FSRCNN':
        generator = generator_FSRCNN(scale_factor=1)
    else:
        raise NotImplementedError(
            "Generator architecture '{}' is not recognized or implemented".format(config.generator))

    discriminator = Discriminator(input_shape=(1, att_config.patch_size, att_config.patch_size))
    feature_extractor = FeatureExtractor()

    if args.source == 'sim':
        img = SimImage(number=args.num,
                       middle_slices=None,
                       every_other=1,
                       data_resolution=att_config.data_resolution,
                       augment=False)
    elif args.source == 'hcp':
        img = HCPImage(number=args.num,
                       middle_slices=None,
                       every_other=1,
                       augment=False)
    # elif args.source == 'hcp_gen':
    #     img = HCPImageGen(number=args.num,
    #                       middle_slices=None,
    #                       every_other=1,
    #                       augment=False)
    elif args.source == 'mrbrains18':
        img = MRBrainS18Image(number=args.num,
                              middle_slices=None,
                              every_other=1)
    elif args.source == 'oasis':
        img = OASISImage(number=args.num,
                         middle_slices=None,
                         every_other=1)
    else:
        raise ValueError("Source '{}' not recognized".format(args.source))
    subject = img.subject()

    path = os.path.join(args.root_dir, ckpt_path)

    overlap, nr_patches = calculate_overlap(subject,
                                            (att_config.patch_size, att_config.patch_size),
                                            (att_config.patch_overlap, att_config.patch_overlap))

    test_set = tio.SubjectsDataset(
        [subject], transform=Normalize(std=args.std))

    grid_sampler = tio.inference.GridSampler(
        test_set[0],
        patch_size=(att_config.patch_size, att_config.patch_size, 1),
        patch_overlap=overlap,
        padding_mode=0,
    )

    if args.gan:
        model = LitTrainer_gan.load_from_checkpoint(
            netG=generator,
            netF=feature_extractor,
            netD=discriminator,
            checkpoint_path=path,
            config=att_config,
            args=args
        )
    else:
        model = LitTrainer_org.load_from_checkpoint(
            netG=generator,
            netF=feature_extractor,
            checkpoint_path=path,
            config=att_config,
            args=args
        )

    model.to(device)
    model.eval()

    aggregator = tio.inference.GridAggregator(grid_sampler)  # , overlap_mode='average')

    patch_loader = torch.utils.data.DataLoader(
        grid_sampler, batch_size=att_config.batch_size)

    start_time = time.time()

    with torch.no_grad():
        for patches_batch in patch_loader:
            if att_config.data_resolution == '2mm_1mm' or args.source == 'mrbrains18':
                imgs_hr = patches_batch['HR'][tio.DATA].squeeze(4)
                imgs_sr = model(imgs_hr.to(device)).unsqueeze(4)
            else:
                imgs_lr = patches_batch['LR'][tio.DATA].squeeze(4)
                imgs_sr = model(imgs_lr.to(device)).unsqueeze(4)

            locations = patches_batch[tio.LOCATION]
            aggregator.add_batch(imgs_sr, locations)

    end_time = time.time()
    print('Time: {:.10f} s'.format(end_time - start_time))

    foreground = aggregator.get_output_tensor() * args.std

    generated = tio.ScalarImage(tensor=foreground)
    subject.add_image(generated, 'SR')

    header = img.info()['LR']['header']
    max_vals = {
        'LR': img.info()['LR']['scaling'],
        'SR': img.info()['LR']['scaling'],
    }

    if args.source == 'sim' or args.source == 'hcp' or args.source == 'mrbrains18':
        max_vals['HR'] = img.info()['HR']['scaling']

    # output_path = 'output/' + os.path.split(os.path.split(ckpt_path)[0])[1]
    output_path = 'output/experiments'

    save_subject(subject=subject,
                 header=header,
                 max_vals=max_vals,
                 pref=args.name,
                 path=output_path,
                 source=args.source
                 )


if __name__ == '__main__':
    main(default_config, ckpt_path)
