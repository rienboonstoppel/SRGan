import os
import torch
import torchio as tio
from argparse import ArgumentParser
from dataset_tio import sim_data, MRBrainS18_data, HCP_data, OASIS_data, perc_norm
from metrics import get_scores
from tqdm import tqdm
import nibabel as nib
import numpy as np
import pandas as pd

def main():
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
        fname = '_3T_T1w_MPR1_img_SR_'
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

    nr_hcp_train = [1, 2]
    nr_sim_train = [1, 2, 5]

    for i in tqdm(range(len(val_subjects))):
        subject = val_subjects[i]
        for nr_hcp in nr_hcp_train:
            for nr_sim in nr_sim_train:
                folder = 'hcp{:02d}_sim{:02d}'.format(nr_hcp, nr_sim)
                SR_path = os.path.join(args.root_dir, 'output', 'sweep-2', args.source, folder, 'SR')

                _fname = '{:06d}'.format(subjects_info[i]['id']) + fname + 'hcp{:02d}_sim{:02d}.nii.gz'.format(nr_hcp,
                                                                                                               nr_sim)
                os.path.join(SR_path, _fname)
                SR = nib.load(os.path.join(SR_path, _fname))
                SR_norm, _ = perc_norm(SR.get_fdata())
                subject.add_image(tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims(SR_norm, 0))),
                                  'SR_{}_{}'.format(nr_hcp, nr_sim))

    ssim_df = pd.DataFrame(columns=[1, 2],
                           index=[1, 2, 5])
    ssim_df = ssim_df.apply(lambda s: s.fillna({j: [] for j in ssim_df.index}))

    ncc_df = pd.DataFrame(columns=[1, 2],
                          index=[1, 2, 5])
    ncc_df = ncc_df.apply(lambda s: s.fillna({j: [] for j in ncc_df.index}))

    nrmse_df = pd.DataFrame(columns=[1, 2],
                            index=[1, 2, 5])
    nrmse_df = nrmse_df.apply(lambda s: s.fillna({j: [] for j in nrmse_df.index}))

    #TODO implement post_proc
    for i in tqdm(range(len(val_subjects))):
        subject = val_subjects[i]
        for nr_hcp in nr_hcp_train:
            for nr_sim in nr_sim_train:
                SR = subject['SR_{}_{}'.format(nr_hcp, nr_sim)]
                HR = subject['HR']
                mask = subject['MSK'].data
                bg_idx = np.where(mask == 0)
                brain_idx = np.where(mask.squeeze(0) != 0)
                crop_coords = ([brain_idx[i].min() for i in range(len(brain_idx))],
                               [brain_idx[i].max() for i in range(len(brain_idx))])

                ncc, ssim, nrmse = get_scores(HR, SR)
                ssim_df[nr_hcp][nr_sim].append(ssim)
                ncc_df[nr_hcp][nr_sim].append(ncc)
                nrmse_df[nr_hcp][nr_sim].append(nrmse)


if __name__ == '__main__':
    main()