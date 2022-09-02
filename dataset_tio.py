import os
from abc import abstractmethod
from os import path
import numpy as np
import nibabel as nib
from tqdm import tqdm
from torch.utils.data import Dataset
from glob import glob
import random
import torch
import torchio as tio
from torchio.data.subject import Subject
from torchio.transforms.intensity_transform import IntensityTransform
from torchio.transforms.augmentation.random_transform import RandomTransform
import cv2
from skimage import exposure
from skimage import filters
from utils import square_mask, cuboid_mask
from torchio.typing import TypeRangeFloat
from collections import defaultdict
from typing import Tuple
from scipy import signal


def perc_norm(img3d, perc=95):
    img3d = np.clip(img3d - img3d[0, 0, 0], 0, None)
    max_val = np.percentile(img3d, perc)
    img_norm = img3d.astype(float) / max_val.astype(np.float32)
    return img_norm, max_val


def select_slices(img, middle_slices, every_other=1):
    diff = (img.shape[2] - middle_slices) / 2
    img = img[:, :, int(np.ceil(diff)):img.shape[2] - int(np.floor(diff)):every_other]
    return img


def augment_hcp(img3d):
    ## dummy
    img3d_aug = img3d

    ### simple augments
    # unsharp masking
    # img3d_aug = img3d + .5 * (img3d - filters.gaussian(img3d, sigma=(1,1,0), preserve_range=True))

    # gamma = 1.5
    # img3d_aug = exposure.adjust_gamma(img3d, gamma=gamma, gain=1)
    return img3d_aug


class Image(object):
    def __init__(self, middle_slices, every_other, augment=False):
        self.middle_slices = middle_slices
        self.every_other = every_other
        self.augment = augment

    @abstractmethod
    def fnames(self):
        pass

    def to_nifty(self) -> dict:
        fnames = self.fnames()
        if 'LR' not in fnames.keys():
            raise ValueError('At least the LR is necessary for the data')
        niftys = {key: nib.load(fnames[key]) for key in fnames.keys()}
        return niftys

    def subject(self) -> Subject:
        niftys = self.to_nifty()

        if self.middle_slices is None:
            middle_slices = niftys['LR'].get_fdata().shape[2]
        else:
            middle_slices = self.middle_slices
        imgs_np = {key: select_slices(img=niftys[key].get_fdata(),
                                      middle_slices=middle_slices,
                                      every_other=self.every_other)
                   for key in niftys.keys()}

        imgs_np['LR'], self.scaling_LR = perc_norm(imgs_np['LR'])
        if self.augment:
            imgs_np['LR'] = augment_hcp(imgs_np['LR'])

        if 'HR' in imgs_np.keys():
            imgs_np['HR'], self.scaling_HR = perc_norm(imgs_np['HR'])

        subject = tio.Subject({key: tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims((imgs_np[key]), 0)))
                               for key in imgs_np.keys() if key != 'MSK'})

        if 'MSK' in imgs_np.keys():
            if ((imgs_np['MSK'] == 0) | (imgs_np['MSK'] == 1)).all():
                subject.add_image(tio.LabelMap(tensor=torch.from_numpy(np.expand_dims(imgs_np['MSK'], 0))), 'MSK')
            else:
                imgs_np['MSK'][imgs_np['MSK'] > 0] = 1
                msk = cv2.erode(imgs_np['MSK'], np.ones((10, 10)), iterations=3)
                subject.add_image(tio.LabelMap(tensor=torch.from_numpy(np.expand_dims(msk, 0))), 'MSK')
        return subject

    def info(self) -> dict:
        niftys = self.to_nifty()
        img_info = {
            'LR': {
                'header': niftys['LR'].header,
                'scaling': self.scaling_LR,
            },
        }
        if 'HR' in niftys.keys():
            img_info['HR'] = {
                'header': niftys['HR'].header,
                'scaling': self.scaling_HR,
            }
        return img_info


class SimImage(Image):
    def __init__(self, number, root_dir='data', middle_slices=50, every_other=1, data_resolution='2mm_1mm',
                 augment=False):
        super().__init__(middle_slices, every_other, augment)
        self.data_resolution = data_resolution
        self.path = os.path.join(root_dir, "brain_simulated_t1w_mri", data_resolution)
        if data_resolution == '2mm_1mm':
            self.img_fname = "23-Aug-2021_Ernst_labels_{:06d}_" \
                             "3T_T1w_MPR1_img_act_1_contrast_1".format(number)
        elif data_resolution == '1mm_07mm':
            self.img_fname = "08-Apr-2022_Ernst_labels_{:06d}_" \
                             "3T_T1w_MPR1_img_act_1_contrast_1".format(number)

    def fnames(self) -> dict:
        if self.data_resolution == '2mm_1mm':
            lr_fname = path.join(self.path, 'LR_' + 'img', self.img_fname + "_Res_2_2_2_" + 'img' + ".nii.gz")
            hr_fname = path.join(self.path, 'HR_' + 'img', self.img_fname + "_Res_1_1_2_" + 'img' + ".nii.gz")
            msk_fname = path.join(self.path, 'HR_' + 'msk', self.img_fname + "_Res_1_1_2_" + 'msk' + ".nii.gz")
        elif self.data_resolution == '1mm_07mm':
            lr_fname = path.join(self.path, 'LR_' + 'img', self.img_fname + "_Res_1_1_1_" + 'img' + ".nii.gz")
            hr_fname = path.join(self.path, 'HR_' + 'img', self.img_fname + "_Res_0.7_0.7_1_" + 'img' + ".nii.gz")
            msk_fname = path.join(self.path, 'HR_' + 'msk', self.img_fname + "_Res_0.7_0.7_1_" + 'msk' + ".nii.gz")
        else:
            raise ValueError(
                "Resolution '{}' not recognized or available, choose '2mm_1mm' or '1mm_07mm' instead".format(
                    self.data_resolution))
        return {'LR': lr_fname,
                'HR': hr_fname,
                'MSK': msk_fname}


class MRBrainS18Image(Image):
    def __init__(self, number, root_dir='data', middle_slices=50, every_other=1, augment=False):
        super().__init__(middle_slices, every_other, augment)
        self.path = os.path.join(root_dir, 'brain_real_t1w_mri', 'MRBrainS18')
        self.img_fname = "p{:01d}_reg_T1".format(number)
        self.msk_fname = "p{:01d}_segm".format(number)

    def fnames(self) -> dict:
        # lr_fname = path.join(self.path, 'LR', self.img_fname + ".nii.gz")
        lr_fname = path.join(self.path, 'GT', self.img_fname + ".nii.gz")
        msk_fname = path.join(self.path, 'MSK', self.msk_fname + ".nii.gz")
        return {'LR': lr_fname,
                # 'HR': hr_fname,
                'MSK': msk_fname}

class HCPImage(Image):
    def __init__(self, number, root_dir='data', middle_slices=50, every_other=1, augment=False):
        super().__init__(middle_slices, every_other, augment)
        self.path = os.path.join(root_dir, 'brain_real_t1w_mri', 'HCP')
        self.img_fname = "{:01d}_3T_T1w_MPR1_img".format(number)
        self.msk_fname = "labels_{:01d}_3T_T1w_MPR1_img".format(number)

    def fnames(self) -> dict:
        lr_fname = path.join(self.path, 'LR', self.img_fname + ".nii.gz")
        hr_fname = path.join(self.path, 'HR', self.img_fname + ".nii.gz")
        msk_fname = path.join(self.path, 'MSK', self.msk_fname + ".nii.gz")
        return {'LR': lr_fname,
                'HR': hr_fname,
                'MSK': msk_fname}


class OASISImage(Image):
    def __init__(self, number, root_dir='data', middle_slices=50, every_other=1, augment=False):
        super().__init__(middle_slices, every_other, augment)
        self.path = os.path.join(root_dir, 'brain_real_t1w_mri', 'OASIS')
        self.img_fname = "OAS1_{:04d}_MR1_mpr_n4_anon_111_t88_masked_gfc".format(number)

    def fnames(self) -> dict:
        lr_fname = path.join(self.path, 'LR', self.img_fname + ".nii.gz")
        msk_fname = path.join(self.path, 'MSK', self.img_fname + "_fseg.nii.gz")
        return {'LR': lr_fname,
                'MSK': msk_fname}


def sim_data(dataset,
             data_resolution='1mm_07mm',
             nr_train_patients=30,
             nr_val_patients=10,
             nr_test_patients=10,
             middle_slices=50,
             every_other=1,
             root_dir='data',
             randomseed=21011998):
    # define paths
    random.seed(randomseed)
    if data_resolution == '2mm_1mm':
        path = os.path.join(root_dir, "brain_simulated_t1w_mri", data_resolution, 'HR_img/')
        fnames = glob(path + "*.nii.gz")
        ids = sorted(list(map(int, [(fnames[i][-60:-54]) for i in range(len(fnames))])))
    elif data_resolution == '1mm_07mm':
        path = os.path.join(root_dir, "brain_simulated_t1w_mri", data_resolution, 'HR_img/')
        fnames = glob(path + "*.nii.gz")
        ids = sorted(list(map(int, [(fnames[i][-64:-58]) for i in range(len(fnames))])))
    random.shuffle(ids)

    if nr_train_patients + nr_val_patients + nr_test_patients > 200:
        raise ValueError("Total number of patients should be 200 or less")

    if dataset == 'training':
        ids_split = ids[:nr_train_patients]
    elif dataset == 'validation':
        ids_split = ids[-nr_val_patients-nr_test_patients:-nr_test_patients]
    elif dataset == 'test':
        ids_split = ids[-nr_test_patients:]
    else:
        raise ValueError("Dataset '{}' not recognized, use 'training, 'validation' or 'test' instead".format(dataset))

    # make arrays
    subjects = []
    print('Loading simulated {} set...'.format(dataset))
    # for num in tqdm(ids_split, desc='Load {} set\t'.format(dataset), bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}',
    #                 leave=True, position=0):
    for num in ids_split:
        data = SimImage(num, root_dir, middle_slices, every_other, data_resolution)
        subjects.append(data.subject())
    return subjects


def HCP_data(dataset,
             root_dir='data',
             middle_slices=50,
             every_other=1,
             nr_train_patients=30,
             nr_val_patients=10,
             nr_test_patients=10,
             ):
    path = root_dir + "/brain_real_t1w_mri/HCP/HR/"
    fnames = glob(path + "*.nii.gz")
    ids = sorted(list(map(int, [(fnames[i][-29:-23]) for i in range(len(fnames))])))
    random.shuffle(ids)

    if nr_train_patients + nr_val_patients + nr_test_patients > 50:
        raise ValueError("Total number of patients should be 50 or less")

    if dataset == 'training':
        ids_split = ids[:nr_train_patients]
    elif dataset == 'validation':
        ids_split = ids[-nr_val_patients-nr_test_patients:-nr_test_patients]
    elif dataset == 'test':
        ids_split = ids[-nr_test_patients:]
    else:
        raise ValueError("Dataset '{}' not recognized, use 'training, 'validation' or 'test' instead".format(dataset))

    # make arrays
    subjects = []
    print('Loading HCP {} set...'.format(dataset))

    for num in ids_split:
        data = HCPImage(num, root_dir=root_dir, middle_slices=middle_slices, every_other=every_other)
        subjects.append(data.subject())
    return subjects


def MRBrainS18_data(dataset,
                    root_dir='data',
                    middle_slices=50,
                    every_other=1):
    path = root_dir + "/brain_real_t1w_mri/MRBrainS18/GT/"
    fnames = glob(path + "*.nii.gz")
    ids = sorted(list(map(int, [(fnames[i][-15:-14]) for i in range(len(fnames))])))
    # make arrays
    subjects = []
    print('Loading MRBrainS18 dataset...')

    if dataset == 'validation':
        ids_split = ids[:3]
    elif dataset == 'test':
        ids_split = ids[3:]
    else:
        raise ValueError("Dataset '{}' not recognized, use 'validation' or 'test' instead".format(dataset))

    for num in ids_split:
        data = MRBrainS18Image(num, root_dir=root_dir, middle_slices=middle_slices, every_other=every_other)
        subjects.append(data.subject())
    return subjects


def OASIS_data(dataset,
               root_dir='data',
               middle_slices=50,
               every_other=1):
    path = root_dir + "/brain_real_t1w_mri/OASIS/LR/"
    fnames = glob(path + "*.nii.gz")
    ids = sorted(list(map(int, [(fnames[i][-46:-42]) for i in range(len(fnames))])))
    # make arrays
    subjects = []
    print('Loading OASIS dataset...')

    if dataset == 'validation':
        ids_split = ids[:5]
    elif dataset == 'test':
        ids_split = ids[5:]
    else:
        raise ValueError("Dataset '{}' not recognized, use 'validation' or 'test' instead".format(dataset))

    for num in ids_split:
        data = OASISImage(num, root_dir=root_dir, middle_slices=middle_slices, every_other=every_other)
        subjects.append(data.subject())
    return subjects


def data(dataset, nr_hcp=30, nr_sim=30, middle_slices=None, root_dir='data', every_other=1):
    subjects = []
    if nr_hcp != 0:
        hcp_subjects = HCP_data(dataset=dataset, nr_train_patients=nr_hcp, middle_slices=middle_slices,
                                root_dir=root_dir, every_other=every_other)
        subjects.extend(hcp_subjects)
        print('HCP {} dataset with length {}'.format(dataset, len(hcp_subjects)))
    if nr_sim != 0:
        sim_subjects = sim_data(dataset=dataset, nr_train_patients=nr_sim, middle_slices=middle_slices,
                                root_dir=root_dir, every_other=every_other)
        subjects.extend(sim_subjects)
        print('Sim {} dataset with length {}'.format(dataset, len(sim_subjects)))

    random.shuffle(subjects)
    return subjects


def calculate_overlap(img, patch_size, ovl_perc):
    patch_size = np.array([patch_size[0], patch_size[1]])
    ovl_perc = np.array([ovl_perc[0], ovl_perc[1]])
    size = img.shape
    sizeXY = np.array([size[1], size[2]])
    nr_patches = np.divide(sizeXY, ovl_perc * patch_size).astype(int) - 1
    residual = sizeXY - (patch_size + (nr_patches - 1) * ovl_perc * patch_size)
    overlap = (patch_size * np.array(ovl_perc) + np.ceil(np.divide(residual, nr_patches))).astype(int)
    for i in range(len(overlap)):
        overlap[i] = overlap[i] + 1 if overlap[i] % 2 == 1 else overlap[i]
    return (*overlap, 0), nr_patches[0] * nr_patches[1] * size[3]

### old

# def augment_hcp(img3d):
#     ### dummy
#     # img3d_aug = img3d
#
#     ### simple augments
#     # img3d_aug = img3d + 5 * (img3d - filters.gaussian(img3d, sigma=(5,5,0), preserve_range=True))
#     # img3d_aug = img3d
#     # gamma = 1.5
#     # img3d_aug = exposure.adjust_gamma(img3d, gamma=gamma, gain=1)
#
#     ### adding in img domain
#     # size = (170, 250)
#     # mask = cuboid_mask(img3d, size)
#     # mask_gauss = filters.gaussian(mask, sigma=5)
#     #
#     # LR_fft_vol = np.fft.fftshift(np.fft.fftn(img3d))
#     # LR_fft_masked_vol = LR_fft_vol * mask_gauss
#     # LR_highpass_vol = np.abs(np.fft.ifftn(LR_fft_masked_vol))
#     # img3d_aug = img3d + LR_highpass_vol * 5
#
#     ### adding in fft domain
#     # size = (100, 200)
#     # mask = cuboid_mask(img3d, size)
#     # mask_gauss = filters.gaussian(mask, sigma=5)
#     #
#     # LR_fft_vol = np.fft.fftshift(np.fft.fftn(img3d))
#     # LR_fft_masked_vol = LR_fft_vol * mask_gauss
#     # LR_fft_sharpened = LR_fft_vol + LR_fft_masked_vol * .5
#     # img3d_aug = np.abs(np.fft.ifftn(LR_fft_sharpened))
#
#     ### adding with tukey
#     window = signal.tukey(img3d.shape[0], alpha=0.5)
#     window2d = np.repeat(window[:, np.newaxis], img3d.shape[1], axis=1)
#     mask = (1 - (np.rot90(window2d) * window2d)) * 30
#     mask3d = np.repeat(mask[:, :, np.newaxis], img3d.shape[2], axis=2)
#     fourier = np.fft.fftshift(np.fft.fftn(img3d))
#     added = fourier + mask3d * fourier
#     img3d_aug = np.abs(np.fft.ifftn(np.fft.ifftshift(added)))
#
#     return img3d_aug
#
# def create_2d_fft_mask(HR_slice: np.array, padding=60, alpha1=0.5, alpha2=0.2, padding_value=0.4) -> np.array:
#     # make LR filter
#     x = signal.tukey(HR_slice.shape[0] - 2 * padding, alpha1)  # as per image size and desired window size
#     y = signal.tukey(HR_slice.shape[0] - 2 * padding, alpha1)
#     [mask_x, mask_y] = np.meshgrid(x, y)
#
#     # create mask with correct scaling
#     mask_xy = (mask_x * mask_y) * (1 - padding_value) + padding_value
#
#     # keep high freq
#     high_freq_x = signal.tukey(HR_slice.shape[0], alpha2)
#     high_freq_y = signal.tukey(HR_slice.shape[0], alpha2)
#     [high_freqs_x, high_freqs_y] = np.meshgrid(high_freq_x, high_freq_y)
#
#     # create mask with correct scaling
#     high_freqs_xy = (1 - (high_freqs_x * high_freqs_y)) * ((1 - padding_value) / 2)
#
#     # combine masks with correct padding values
#     mask = np.pad(mask_xy, padding, constant_values=padding_value) + high_freqs_xy  # for simple zero padding
#
#     return mask
#
# def create_2d_fft_mask_2(HR_slice: np.array, downscale_factor = 0.7, alpha=0.5) -> np.array:
#     padding = int((HR_slice.shape[0] - (1 / (1 - .5 * alpha)) * HR_slice.shape[0] * downscale_factor) / 2)
#
#     x = signal.tukey(HR_slice.shape[0] - 2 * padding, alpha)
#     y = signal.tukey(HR_slice.shape[0] - 2 * padding, alpha)
#     [mask_x, mask_y] = np.meshgrid(x, y)
#
#     mask = mask_x * mask_y
#
#     mask_padded = np.pad(mask, padding, constant_values=0)
#
#     return mask_padded
#
# def generate_LR(HR: np.array) -> np.array:
#     LR = np.zeros_like(HR)
#     mask = create_2d_fft_mask_2(HR[:, :, 0])
#     for i in range(HR.shape[2]):
#         # go to kspace for HR
#         kspace = np.fft.fftshift(np.fft.fft2(HR[:, :, i]))
#         # applying the filter to HR k-space
#         kspace_filtered = mask * kspace
#         # go back to image domain
#         LR[:, :, i] = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_filtered)))
#     return LR
#
#
# class ImageGen(object):
#     def __init__(self, middle_slices, every_other, hist_eq=False, augment=False):
#         self.middle_slices = middle_slices
#         self.every_other = every_other
#         self.hist_eq = hist_eq
#
#     @abstractmethod
#     def fnames(self):
#         pass
#
#     def to_nifty(self) -> dict:
#         fnames = self.fnames()
#         if 'HR' not in fnames.keys():
#             raise ValueError('At least the HR is necessary for this method')
#         niftys = {key: nib.load(fnames[key]) for key in fnames.keys()}
#         return niftys
#
#     def subject(self) -> Subject:
#         niftys = self.to_nifty()
#
#         if self.middle_slices is None:
#             middle_slices = niftys['HR'].get_fdata().shape[2]
#         else:
#             middle_slices = self.middle_slices
#
#         imgs_np = {key: select_slices(img=niftys[key].get_fdata(),
#                                       middle_slices=middle_slices,
#                                       every_other=self.every_other)
#                    for key in niftys.keys()}
#
#         imgs_np['HR'], self.scaling_HR = perc_norm(imgs_np['HR'])
#
#         imgs_np['LR'] = generate_LR(imgs_np['HR'])
#
#         subject = tio.Subject({key: tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims((imgs_np[key]), 0)))
#                                for key in imgs_np.keys() if key!='MSK'})
#
#         if 'MSK' in imgs_np.keys():
#             if ((imgs_np['MSK']==0) | (imgs_np['MSK']==1)).all():
#                 subject.add_image(tio.LabelMap(tensor=torch.from_numpy(np.expand_dims(imgs_np['MSK'], 0))), 'MSK')
#             else:
#                 imgs_np['MSK'][imgs_np['MSK'] > 0] = 1
#                 msk = cv2.erode(imgs_np['MSK'], np.ones((10, 10)), iterations=3)
#                 subject.add_image(tio.LabelMap(tensor=torch.from_numpy(np.expand_dims(msk, 0))), 'MSK')
#         return subject
#
#     def info(self) -> dict:
#         niftys = self.to_nifty()
#         img_info = {'HR': {
#             'header': niftys['HR'].header,
#             'scaling': self.scaling_HR,
#         }, 'LR': {
#             'header': niftys['HR'].header,
#             'scaling': self.scaling_HR,
#         }}
#         return img_info
# class HCPImageGen(ImageGen):
#     def __init__(self, number, root_dir='data', middle_slices=50, every_other=1, hist_eq=False, augment=False):
#         super().__init__(middle_slices, every_other, hist_eq, augment)
#         self.path = os.path.join(root_dir, 'brain_real_t1w_mri', 'HCP')
#         self.img_fname = "{:01d}_3T_T1w_MPR1_img".format(number)
#         self.msk_fname = "labels_{:01d}_3T_T1w_MPR1_img".format(number)
#
#     def fnames(self) -> dict:
#         hr_fname = path.join(self.path, 'HR', self.img_fname + ".nii.gz")
#         msk_fname = path.join(self.path, 'MSK', self.msk_fname + ".nii.gz")
#         return {'HR': hr_fname,
#                 'MSK': msk_fname}
