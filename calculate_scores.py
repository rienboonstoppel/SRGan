import os
import torch
import torchio as tio
from argparse import ArgumentParser
from dataset_tio import sim_data, HCP_data, perc_norm
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import normalized_root_mse as NRMSE
from skimage.metrics import peak_signal_noise_ratio as PSNR
from tqdm import tqdm
import nibabel as nib
import numpy as np
import copy


def post_proc(img: torch.Tensor, bg_idx: np.ndarray, crop_coords: tuple) -> np.ndarray:
    img_np = copy.deepcopy(img)
    img_np = img_np.data.squeeze().numpy()
    img_np[bg_idx] = 0
    min, max = crop_coords
    img_np = img_np[min[0]:max[0] + 1, min[1]:max[1] + 1, min[2]:max[2] + 1]
    return img_np


project = 'example'
exp_name = 'mixed-wgan'

SRs_path = os.path.join('output', project, exp_name)


def main(SRs_path):
    parser = ArgumentParser()
    parser.add_argument('--root_dir', default='/mnt/beta/djboonstoppel/Code', type=str)
    parser.add_argument('--source', required=True, type=str, choices=['sim', 'hcp'])

    args = parser.parse_args()

    data_path = os.path.join(args.root_dir, 'data')

    args.middle_slices = None
    args.every_other = 1

    dataset = 'test'

    if args.source == 'sim':
        val_subjects, subjects_info = sim_data(dataset=dataset,
                                               root_dir=data_path,
                                               middle_slices=args.middle_slices,
                                               every_other=args.every_other)
    elif args.source == 'hcp':
        val_subjects, subjects_info = HCP_data(dataset=dataset,
                                               root_dir=data_path,
                                               middle_slices=args.middle_slices,
                                               every_other=args.every_other)
    else:
        raise ValueError("Dataset '{}' not implemented".format(args.source))

    SRs_path = os.path.join(SRs_path, args.source, dataset)
    for i in tqdm(range(len(val_subjects)), desc='Adding SR images'):
        subject = val_subjects[i]
        if args.source == 'sim':
            fname = '08-Apr-2022_Ernst_labels_{:06d}_3T_T1w_MPR1_img_act_1_contrast_1_SR.nii.gz'.format(
                subjects_info[i]['id'])
        elif args.source == 'hcp':
            fname = '{:06d}_3T_T1w_MPR1_img_SR.nii.gz'.format(subjects_info[i]['id'])
        path = os.path.join(SRs_path, fname)
        SR = nib.load(path)
        SR_norm, _ = perc_norm(SR.get_fdata())
        subject.add_image(tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims(SR_norm, 0))), 'SR')

    ssims = []
    nrmses = []
    psnrs = []
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

        SR = post_proc(img=subject['SR'],
                       bg_idx=bg_idx,
                       crop_coords=crop_coords)

        ssims.append(
            SSIM(HR, SR, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.5))
        nrmses.append(NRMSE(HR, SR))
        psnrs.append(PSNR(HR, SR, data_range=1.5))

    print('---')
    print('{:<10}| {:<10}'.format('Metric', 'mean ± std'))
    print('{:<10}| {:.4f} ± {:.4f}'.format('SSIM ↑', np.mean(ssims), np.std(ssims)))
    print('{:<10}| {:.2f} ± {:.4f}'.format('PSNR ↑', np.mean(psnrs), np.std(psnrs)))
    print('{:<10}| {:.5f} ± {:.5f}'.format('NRMSE ↓', np.mean(nrmses), np.std(nrmses)))
    print('---')


if __name__ == '__main__':
    main(SRs_path)
