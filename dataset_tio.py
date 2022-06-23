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
from torchio.typing import TypeRangeFloat
from collections import defaultdict
from typing import Tuple


def perc_norm(img3d, perc=95):
    max_val = np.percentile(img3d, perc)
    img_norm = img3d.astype(float) / max_val.astype(np.float32)
    return img_norm, max_val


def select_slices(img, middle_slices, every_other=1):
    diff = (img.shape[2] - middle_slices) / 2
    img = img[:, :, int(np.ceil(diff)):img.shape[2] - int(np.floor(diff)):every_other]
    return img


class Image(object):
    def __init__(self, middle_slices, every_other, hist_eq=False):
        self.middle_slices = middle_slices
        self.every_other = every_other
        self.hist_eq = hist_eq

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

        imgs_np = {key: select_slices(img=niftys[key].get_fdata()-niftys[key].get_fdata()[0,0,0],
                                         middle_slices=middle_slices,
                                         every_other=self.every_other)
                      for key in niftys.keys()}

        imgs_np['LR'], self.scaling_LR = perc_norm(imgs_np['LR'])
        # a, self.scaling_LR = perc_norm(imgs_np['LR'])

        # img_cdf, bin_centers = exposure.cumulative_distribution(imgs_np['LR'])
        # imgs_np['LR'] = np.interp(imgs_np['LR'], bin_centers, img_cdf)
        # imgs_np['LR'] = imgs_np['LR'] - imgs_np['LR'][0,0,0]

        # imgs_np['LR'] = exposure.equalize_adapthist(np.clip(imgs_np['LR'], 0, 1))

        if self.hist_eq:
            imgs_np['LR'] = exposure.equalize_hist(imgs_np['LR'], mask=imgs_np['MSK'])

        if 'HR' in imgs_np.keys():
            imgs_np['HR'], self.scaling_HR = perc_norm(imgs_np['HR'])
            a, self.scaling_HR = perc_norm(imgs_np['HR'])
            if self.hist_eq:
                imgs_np['HR'] = exposure.equalize_hist(imgs_np['HR'], mask=imgs_np['MSK'])

        subject = tio.Subject({key: tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims((imgs_np[key]), 0)))
                               for key in imgs_np.keys() if key!='MSK'})

        if 'MSK' in imgs_np.keys():
            if ((imgs_np['MSK']==0) | (imgs_np['MSK']==1)).all():
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
    def __init__(self, number, root_dir='data', middle_slices=50, every_other=1, data_resolution='2mm_1mm', hist_eq=False):
        super().__init__(middle_slices, every_other, hist_eq)
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
            raise ValueError("Resolution '{}' not recognized or available, choose '2mm_1mm' or '1mm_07mm' instead".format(self.data_resolution))
        return {'LR': lr_fname,
                'HR': hr_fname,
                'MSK': msk_fname}

class MRBrainS18Image(Image):
    def __init__(self, number, root_dir='data', middle_slices=50, every_other=1, hist_eq=False):
        super().__init__(middle_slices, every_other, hist_eq)
        self.path = os.path.join(root_dir, 'brain_real_t1w_mri', 'MRBrainS18')
        self.img_fname = "p{:01d}_reg_T1".format(number)
        self.msk_fname = "p{:01d}_segm".format(number)

    def fnames(self) -> dict:
        lr_fname = path.join(self.path, 'LR', self.img_fname + ".nii.gz")
        hr_fname = path.join(self.path, 'GT', self.img_fname + ".nii.gz")
        msk_fname = path.join(self.path, 'MSK', self.msk_fname + ".nii.gz")
        return {'LR': lr_fname,
                'HR': hr_fname,
                'MSK': msk_fname}

class HCPImage(Image):
    def __init__(self, number, root_dir='data', middle_slices=50, every_other=1, hist_eq=False):
        super().__init__(middle_slices, every_other, hist_eq)
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
    def __init__(self, number, root_dir='data', middle_slices=50, every_other=1, hist_eq=False):
        super().__init__(middle_slices, every_other, hist_eq)
        self.path = os.path.join(root_dir, 'brain_real_t1w_mri', 'OASIS')
        self.img_fname = "OAS1_{:04d}_MR1_mpr_n4_anon_111_t88_masked_gfc".format(number)

    def fnames(self) -> dict:
        lr_fname = path.join(self.path, 'LR', self.img_fname + ".nii.gz")
        msk_fname = path.join(self.path, 'MSK', self.img_fname + "_fseg.nii.gz")
        return {'LR': lr_fname,
                'MSK': msk_fname}

def sim_data(dataset,
             data_resolution='1mm_07mm',
             patients_frac=1,
             train_frac=0.7,
             val_frac=.15,
             test_frac=.15,
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
    # random.shuffle(ids)

    # define data splits
    split1 = int(np.floor(patients_frac * train_frac * len(ids)))
    split2 = split1 + int(np.floor(patients_frac * val_frac * len(ids)))
    split3 = split2 + int(np.floor(patients_frac * test_frac * len(ids)))

    if dataset == 'training':
        ids_split = ids[:split1]
    elif dataset == 'validation':
        ids_split = ids[split1:split2]
    elif dataset == 'test':
        ids_split = ids[split2:split3]
    else: raise ValueError("Dataset '{}' not recognized, use 'training, 'validation' or 'test' instead".format(dataset))

    # make arrays
    subjects = []
    print('Loading simulated {} set...'.format(dataset))
    # for num in tqdm(ids_split, desc='Load {} set\t'.format(dataset), bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}',
    #                 leave=True, position=0):
    for num in ids_split:
        data = SimImage(num, root_dir, middle_slices, every_other, data_resolution)
        subjects.append(data.subject())
    return subjects


def MRBrainS18_data(root_dir='data', middle_slices=50, every_other=1):
    path = root_dir + "/brain_real_t1w_mri/MRBrainS18/GT/"
    fnames = glob(path + "*.nii.gz")
    ids = sorted(list(map(int, [(fnames[i][-15:-14]) for i in range(len(fnames))])))
    # make arrays
    subjects = []
    print('Loading MRBrainS18 dataset...')

    for num in ids:
        data = MRBrainS18Image(num, root_dir=root_dir, middle_slices=middle_slices, every_other=every_other)
        subjects.append(data.subject())
    return subjects


def HCP_data(dataset,
             root_dir='data',
             middle_slices=50,
             every_other=1,
             patients_frac=1,
             train_frac=0.7,
             val_frac=.15,
             test_frac=.15,
             ):
    path = root_dir + "/brain_real_t1w_mri/HCP/HR/"
    fnames = glob(path + "*.nii.gz")
    ids = sorted(list(map(int, [(fnames[i][-29:-23]) for i in range(len(fnames))])))

    # define data splits
    split1 = int(np.floor(patients_frac * train_frac * len(ids)))
    split2 = split1 + int(np.floor(patients_frac * val_frac * len(ids)))
    split3 = split2 + int(np.floor(patients_frac * test_frac * len(ids)))

    if dataset == 'training':
        ids_split = ids[:split1]
    elif dataset == 'validation':
        ids_split = ids[split1:split2]
    elif dataset == 'test':
        ids_split = ids[split2:split3]
    else: raise ValueError("Dataset '{}' not recognized, use 'training, 'validation' or 'test' instead".format(dataset))

    # make arrays
    subjects = []
    print('Loading HCP {} set...'.format(dataset))

    for num in ids_split:
        data = HCPImage(num, root_dir=root_dir, middle_slices=middle_slices, every_other=every_other)
        subjects.append(data.subject())
    return subjects

def mixed_data(dataset, patients_frac, middle_slices=None, root_dir='data', every_other=1):
    sim_subjects = sim_data(dataset=dataset, patients_frac=patients_frac/4, middle_slices=middle_slices, root_dir=root_dir, every_other=every_other)
    hcp_subjects = HCP_data(dataset=dataset, patients_frac=patients_frac/2, middle_slices=middle_slices, root_dir=root_dir, every_other=every_other)
    mixed = sim_subjects + hcp_subjects
    random.shuffle(mixed)
    return mixed


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




# def real_data(dataset, num_train_patients, numslices=45, root_dir='data'):
#     path = root_dir + "/brain_real_t1w_mri/GT/"
#     fnames = glob(path + "*.nii.gz")
#     ids = sorted(list(map(int, [(fnames[i][-15:-14]) for i in range(len(fnames))])))
#     random.shuffle(ids)
#
#     if dataset == 'training':
#         ids_split = ids[:num_train_patients]
#     elif dataset == 'validation':
#         ids_split = [6]
#     elif dataset == 'test':
#         ids_split = [6]
#
#     # make arrays
#     subjects = []
#     print('Loading real {} set...'.format(dataset))
#
#     for num in ids_split:
#         data = RealImage(num, root_dir=root_dir, select_slices=numslices)
#         subjects.append(data.subject())
#     return subjects
#
#
# def mixed_data(dataset, combined_num_patients, num_real=3, numslices=None, root_dir='data'):
#     sim_subjects = data_split(dataset=dataset, num_patients=combined_num_patients - num_real, numslices=numslices, root_dir=root_dir)
#     real_subjects = real_data(dataset=dataset, num_train_patients=num_real, numslices=numslices, root_dir=root_dir)
#     mixed = sim_subjects + real_subjects
#     random.shuffle(mixed)
#     return mixed
