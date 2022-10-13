import os
import torch
import torchio as tio
from argparse import ArgumentParser
from dataset_tio import sim_data, MRBrainS18_data, HCP_data, OASIS_data, perc_norm
# from utils import NCC
# from skimage.metrics import structural_similarity as SSIM
# from skimage.metrics import normalized_root_mse as NRMSE
from tqdm import tqdm
import nibabel as nib
import numpy as np
import pandas as pd
import copy
from flip_api import compute_ldrflip as flip

def post_proc(img:torch.Tensor, bg_idx:np.ndarray, crop_coords:tuple) -> np.ndarray:
    img_np = copy.deepcopy(img)
    img_np = img_np.data.squeeze().numpy()
    img_np[bg_idx] = 0
    # nonzero_mean = img_np[np.nonzero(img_np)].mean()
    # img_np -= nonzero_mean
    # img_np[bg_idx] = 0
    min, max = crop_coords
    img_np = img_np[min[0]:max[0]+1, min[1]:max[1]+1, min[2]:max[2]+1]
    return img_np


def main():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', default='/mnt/beta/djboonstoppel/Code', type=str)
    parser.add_argument('--source', required=True, type=str, choices=['sim', 'hcp'])

    args = parser.parse_args()

    data_path = os.path.join(args.root_dir, 'data')

    args.middle_slices = None
    args.every_other = 1

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
    else:
        raise ValueError("Dataset '{}' not implemented".format(args.source))

    # nr_hcp_train = [1, 2, 5, 10, 20, 30]
    # nr_sim_train = [1, 2, 5, 10, 20, 30, 50]
    nr_hcp_train = [0]
    nr_sim_train = [50]

    for i in tqdm(range(len(val_subjects)), desc='Adding SR images'):
        subject = val_subjects[i]
        for nr_hcp in nr_hcp_train:
            for nr_sim in nr_sim_train:
                folder = 'hcp{:02d}_sim{:02d}'.format(nr_hcp, nr_sim)
                SR_path = os.path.join(args.root_dir, 'output', 'baseline', args.source, folder, 'SR')
                if args.source == 'sim':
                    fname = '08-Apr-2022_Ernst_labels_{:06d}_3T_T1w_MPR1_img_act_1_contrast_1_SR_hcp{:02d}_sim{:02d}.nii.gz'.format(subjects_info[i]['id'],
                                                                                          nr_hcp, nr_sim)
                elif args.source == 'hcp':
                    fname = '{:06d}_3T_T1w_MPR1_img_SR_hcp{:02d}_sim{:02d}.nii.gz'.format(subjects_info[i]['id'],
                                                                                          nr_hcp, nr_sim)
                os.path.join(SR_path, fname)
                SR = nib.load(os.path.join(SR_path, fname))
                SR_norm, _ = perc_norm(SR.get_fdata())
                subject.add_image(tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims(SR_norm, 0))),
                                  'SR_{}_{}'.format(nr_hcp, nr_sim))

    # ssim_df = pd.DataFrame(columns=nr_hcp_train, index=nr_sim_train)
    # ssim_df = ssim_df.apply(lambda s: s.fillna({i: [] for i in ssim_df.index}))
    #
    # ncc_df = pd.DataFrame(columns=nr_hcp_train, index=nr_sim_train)
    # ncc_df = ncc_df.apply(lambda s: s.fillna({i: [] for i in ncc_df.index}))
    #
    # nrmse_df = pd.DataFrame(columns=nr_hcp_train, index=nr_sim_train)
    # nrmse_df = nrmse_df.apply(lambda s: s.fillna({i: [] for i in nrmse_df.index}))

    flip_mean_df = pd.DataFrame(columns=nr_hcp_train, index=nr_sim_train)
    flip_mean_df = flip_mean_df.apply(lambda s: s.fillna({j: [] for j in flip_mean_df.index}))

    flip_max_df = pd.DataFrame(columns=nr_hcp_train, index=nr_sim_train)
    flip_max_df = flip_max_df.apply(lambda s: s.fillna({j: [] for j in flip_max_df.index}))

    for i in tqdm(range(len(val_subjects)), desc='Calculating metrics'):
        subject = val_subjects[i]
        mask = subject['MSK'].data.squeeze()
        bg_idx = np.where(mask == 0)
        brain_idx = np.where(mask != 0)
        crop_coords = ([brain_idx[i].min() for i in range(len(brain_idx))],
                       [brain_idx[i].max() for i in range(len(brain_idx))])
        HR = post_proc(subject['HR'],
                       bg_idx=bg_idx,
                       crop_coords=crop_coords)
        reference = np.repeat(np.expand_dims(HR[:, :, 100], 0), 3, 0)

        for nr_hcp in nr_hcp_train:
            for nr_sim in nr_sim_train:
                SR = post_proc(img=subject['SR_{}_{}'.format(nr_hcp, nr_sim)],
                               bg_idx=bg_idx,
                               crop_coords=crop_coords)

                test = np.repeat(np.expand_dims(SR[:, :, 100], 0), 3, 0)
                flip_mean, flip_max = flip(reference, test)
                flip_mean_df[nr_hcp][nr_sim].append(flip_mean)
                flip_max_df[nr_hcp][nr_sim].append(flip_max)
                # ssim_df[nr_hcp][nr_sim].append(
                #     SSIM(HR, SR, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.5))
                # ncc_df[nr_hcp][nr_sim].append(NCC(HR, SR))
                # nrmse_df[nr_hcp][nr_sim].append(NRMSE(HR, SR))
    output_path = os.path.join(args.root_dir, 'output', 'baseline', args.source)
    # ssim_df.to_csv(os.path.join(output_path, 'ssim_df_baseline_hcp_new.csv'))
    # ncc_df.to_csv(os.path.join(output_path, 'ncc_df_baseline_hcp_new.csv'))
    # nrmse_df.to_csv(os.path.join(output_path, 'nrmse_df_baseline_hcp_new.csv'))
    flip_mean_df.to_csv(os.path.join(output_path, 'flip_df_mean_baseline_sim.csv'))
    flip_max_df.to_csv(os.path.join(output_path, 'flip_df_max_baseline_sim.csv'))

if __name__ == '__main__':
    main()