import os
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
import cv2


def perc_norm(img3d, perc=95):
    max_val = np.percentile(img3d, perc)
    img_norm = img3d.astype(float) / max_val.astype(np.float32)
    return img_norm, max_val


def select_slices(img, middle_slices, every_other=1, datasource='1mm_07mm'):
    diff = (img.shape[2] - middle_slices) / 2
    img = img[:, :, int(np.ceil(diff)):img.shape[2] - int(np.floor(diff)):every_other]

    # if datasource == '2mm_1mm' or datasource == 'real':
    #     img = img[:, :, int(np.ceil(diff)):img.shape[2] - int(np.floor(diff)):every_other]
    # elif datasource == '1mm_07mm':
    #     img = img[:, :, int(np.ceil(diff)) + 20:img.shape[2] - int(np.floor(diff)) + 20:every_other]
    return img


class SimImagePair(object):
    def __init__(self, number, root_dir='data', middle_slices=50, every_other=1, datasource='2mm_1mm'):
        self._number = number
        self.middle_slices = middle_slices
        self.every_other = every_other
        self.datasource = datasource
        self.path = os.path.join(root_dir, "brain_simulated_t1w_mri", datasource)
        if datasource == '2mm_1mm':
            self.img_fname = "23-Aug-2021_Ernst_labels_{:06d}_" \
                             "3T_T1w_MPR1_img_act_1_contrast_1".format(self._number)
        elif datasource == '1mm_07mm':
            self.img_fname = "08-Apr-2022_Ernst_labels_{:06d}_" \
                             "3T_T1w_MPR1_img_act_1_contrast_1".format(self._number)

    def fnames(self):
        if self.datasource == '2mm_1mm':
            LRf = path.join(self.path, 'LR_' + 'img', self.img_fname + "_Res_2_2_2_" + 'img' + ".nii.gz")
            HRf = path.join(self.path, 'HR_' + 'img', self.img_fname + "_Res_1_1_2_" + 'img' + ".nii.gz")
            MSKf = path.join(self.path, 'HR_' + 'msk', self.img_fname + "_Res_1_1_2_" + 'msk' + ".nii.gz")
        elif self.datasource == '1mm_07mm':
            LRf = path.join(self.path, 'LR_' + 'img', self.img_fname + "_Res_1_1_1_" + 'img' + ".nii.gz")
            HRf = path.join(self.path, 'HR_' + 'img', self.img_fname + "_Res_0.7_0.7_1_" + 'img' + ".nii.gz")
            MSKf = path.join(self.path, 'HR_' + 'msk', self.img_fname + "_Res_0.7_0.7_1_" + 'msk' + ".nii.gz")
        return LRf, HRf, MSKf

    def to_nifty(self):
        LR_fname, HR_fname, MSK_fname = self.fnames()
        self.LR = nib.load(LR_fname)
        self.HR = nib.load(HR_fname)
        self.MSK = nib.load(MSK_fname)

    def subject(self):
        self.to_nifty()
        if self.middle_slices == None:
            middle_slices = self.LR.get_fdata().shape[2]
        else:
            middle_slices = self.middle_slices

        LR = select_slices(img=self.LR.get_fdata(), middle_slices=middle_slices, every_other=self.every_other,
                           datasource=self.datasource)
        HR = select_slices(img=self.HR.get_fdata(), middle_slices=middle_slices, every_other=self.every_other,
                           datasource=self.datasource)
        MSK = select_slices(img=self.MSK.get_fdata(), middle_slices=middle_slices, every_other=self.every_other,
                            datasource=self.datasource)
        MSK[MSK > 0] = 1
        MSK = cv2.erode(MSK, np.ones((10, 10)), iterations=3)

        LR_norm, self.scaling_LR = perc_norm(LR)
        HR_norm, self.scaling_HR = perc_norm(HR)

        subject = tio.Subject(
            LR=tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims(LR_norm, 0))),
            HR=tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims(HR_norm, 0))),
            MSK=tio.LabelMap(tensor=torch.from_numpy(np.expand_dims(MSK, 0))),
        )
        return subject

    def info(self):
        self.to_nifty()
        img_info = {
            'LR': {
                'header': self.LR.header,
                'scaling': self.scaling_LR,
            },
            'HR': {
                'header': self.HR.header,
                'scaling': self.scaling_HR,
            }
        }
        return img_info


class RealImagePair(object):
    def __init__(self, number, root_dir='data', middle_slices=50, every_other=1):
        self._number = number
        self.middle_slices = middle_slices
        self.every_other = every_other
        self.path = root_dir + "/brain_real_t1w_mri"
        self.img_fname = "p{:01d}_reg_T1".format(self._number)
        self.msk_fname = "p{:01d}_segm".format(self._number)

    def fnames(self):
        LRf = path.join(self.path, 'LR', self.img_fname + ".nii.gz")
        HRf = path.join(self.path, 'GT', self.img_fname + ".nii.gz")
        MSKf = path.join(self.path, 'MSK', self.msk_fname + ".nii.gz")
        return LRf, HRf, MSKf

    def to_nifty(self):
        LR_fname, HR_fname, MSK_fname = self.fnames()
        self.LR = nib.load(LR_fname)
        self.HR = nib.load(HR_fname)
        self.MSK = nib.load(MSK_fname)

    def subject(self):
        self.to_nifty()

        if self.middle_slices == None:
            middle_slices = self.LR.get_fdata().shape[2]
        else:
            middle_slices = self.middle_slices

        LR = select_slices(img=self.LR.get_fdata(), middle_slices=middle_slices, every_other=self.every_other,
                           datasource='real')
        HR = select_slices(img=self.HR.get_fdata(), middle_slices=middle_slices, every_other=self.every_other,
                           datasource='real')
        MSK = select_slices(img=self.MSK.get_fdata(), middle_slices=middle_slices, every_other=self.every_other,
                            datasource='real')
        MSK[MSK > 0] = 1
        MSK = cv2.erode(MSK, np.ones((10, 10)), iterations=3)

        LR_norm, self.scaling_LR = perc_norm(LR)
        HR_norm, self.scaling_HR = perc_norm(HR)

        subject = tio.Subject(
            LR=tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims(LR_norm, 0))),
            HR=tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims(HR_norm, 0))),
            MSK=tio.LabelMap(tensor=torch.from_numpy(np.expand_dims(MSK, 0))),
        )
        return subject

    def info(self):
        self.to_nifty()
        img_info = {
            'LR': {
                'header': self.LR.header,
                'scaling': self.scaling_LR,
            },
            'HR': {
                'header': self.HR.header,
                'scaling': self.scaling_HR,
            }
        }
        return img_info


def sim_data(dataset,
             datasource='1mm_07mm',
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
    if datasource == '2mm_1mm':
        path = os.path.join(root_dir, "brain_simulated_t1w_mri", datasource, 'HR_img/')
        fnames = glob(path + "*.nii.gz")
        ids = sorted(list(map(int, [(fnames[i][-60:-54]) for i in range(len(fnames))])))
    elif datasource == '1mm_07mm':
        path = os.path.join(root_dir, "brain_simulated_t1w_mri", datasource, 'HR_img/')
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

    # make arrays
    subjects = []
    print('Loading simulated {} set...'.format(dataset))
    # for num in tqdm(ids_split, desc='Load {} set\t'.format(dataset), bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}',
    #                 leave=True, position=0):
    for num in ids_split:
        data = SimImagePair(num, root_dir, middle_slices, every_other, datasource)
        subjects.append(data.subject())
    return subjects


def real_data(root_dir='data', middle_slices=50, every_other=1):
    path = root_dir + "/brain_real_t1w_mri/GT/"
    fnames = glob(path + "*.nii.gz")
    ids = sorted(list(map(int, [(fnames[i][-15:-14]) for i in range(len(fnames))])))

    # make arrays
    subjects = []
    print('Loading real dataset...')

    for num in ids:
        data = RealImagePair(num, root_dir=root_dir, middle_slices=middle_slices, every_other=every_other)
        subjects.append(data.subject())
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


class Normalize(IntensityTransform):
    def __init__(
            self,
            std,
            mean=0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.std = std
        self.mean = mean

    def apply_transform(self, subject: Subject) -> Subject:
        for image_name, image in self.get_images_dict(subject).items():
            self.apply_normalization(subject, image_name)
        return subject

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
    ) -> None:
        image = subject[image_name]
        standardized = self.znorm(
            image.data,
            self.std,
            self.mean,
        )
        image.set_data(standardized)

    @staticmethod
    def znorm(tensor: torch.Tensor, std, mean) -> torch.Tensor:
        tensor = tensor.clone().float()
        tensor -= mean
        tensor /= std
        return tensor

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
