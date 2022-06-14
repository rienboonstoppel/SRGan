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
import cv2


def perc_norm(img3d, perc=95):
    max_val = np.percentile(img3d, perc)
    img_norm = img3d.astype(float) / max_val.astype(np.float32)
    return img_norm, max_val


def select_slices(img, middle_slices, every_other=1):
    diff = (img.shape[2] - middle_slices) / 2
    img = img[:, :, int(np.ceil(diff)):img.shape[2] - int(np.floor(diff)):every_other]
    return img


class Image(object):
    def __init__(self, middle_slices, every_other):
        self.middle_slices = middle_slices
        self.every_other = every_other

    @abstractmethod
    def fnames(self):
        pass

    def to_nifty(self):
        lr_fname, hr_fname, msk_fname = self.fnames()
        lr = nib.load(lr_fname)
        hr = nib.load(hr_fname)
        msk = nib.load(msk_fname)
        return lr, hr, msk

    def subject(self):
        lr_nib, hr_nib, msk_nib = self.to_nifty()
        if self.middle_slices is None:
            middle_slices = lr_nib.get_fdata().shape[2]
        else:
            middle_slices = self.middle_slices

        lr = select_slices(img=lr_nib.get_fdata(),
                           middle_slices=middle_slices,
                           every_other=self.every_other)
        hr = select_slices(img=hr_nib.get_fdata(),
                           middle_slices=middle_slices,
                           every_other=self.every_other)
        msk = select_slices(img=msk_nib.get_fdata(),
                            middle_slices=middle_slices,
                            every_other=self.every_other)
        if ((msk==0) | (msk==1)).all():
            pass
        else:
            msk[msk > 0] = 1
            msk = cv2.erode(msk, np.ones((10, 10)), iterations=3)

        lr_norm, self.scaling_LR = perc_norm(lr)
        hr_norm, self.scaling_HR = perc_norm(hr)

        subject = tio.Subject(
            LR=tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims(lr_norm, 0))),
            HR=tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims(hr_norm, 0))),
            MSK=tio.LabelMap(tensor=torch.from_numpy(np.expand_dims(msk, 0))),
        )
        return subject

    def info(self):
        lr_nib, hr_nib, msk_nib = self.to_nifty()
        img_info = {
            'LR': {
                'header': lr_nib.header,
                'scaling': self.scaling_LR,
            },
            'HR': {
                'header': hr_nib.header,
                'scaling': self.scaling_HR,
            }
        }
        return img_info


class SimImage(Image):
    def __init__(self, number, root_dir='data', middle_slices=50, every_other=1, data_resolution='2mm_1mm'):
        super().__init__(middle_slices, every_other)
        self.data_resolution = data_resolution
        self.path = os.path.join(root_dir, "brain_simulated_t1w_mri", data_resolution)
        if data_resolution == '2mm_1mm':
            self.img_fname = "23-Aug-2021_Ernst_labels_{:06d}_" \
                             "3T_T1w_MPR1_img_act_1_contrast_1".format(number)
        elif data_resolution == '1mm_07mm':
            self.img_fname = "08-Apr-2022_Ernst_labels_{:06d}_" \
                             "3T_T1w_MPR1_img_act_1_contrast_1".format(number)

    def fnames(self):
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
        return lr_fname, hr_fname, msk_fname

class MRBrainS18Image(Image):
    def __init__(self, number, root_dir='data', middle_slices=50, every_other=1):
        super().__init__(middle_slices, every_other)
        self.path = os.path.join(root_dir, 'brain_real_t1w_mri', 'MRBrainS18')
        self.img_fname = "p{:01d}_reg_T1".format(number)
        self.msk_fname = "p{:01d}_segm".format(number)

    def fnames(self):
        lr_fname = path.join(self.path, 'LR', self.img_fname + ".nii.gz")
        hr_fname = path.join(self.path, 'GT', self.img_fname + ".nii.gz")
        msk_fname = path.join(self.path, 'MSK', self.msk_fname + ".nii.gz")
        return lr_fname, hr_fname, msk_fname

class HCPImage(Image):
    def __init__(self, number, root_dir='data', middle_slices=50, every_other=1):
        super().__init__(middle_slices, every_other)
        self.path = os.path.join(root_dir, 'brain_real_t1w_mri', 'HCP')
        self.img_fname = "{:01d}_3T_T1w_MPR1_img".format(number)
        self.msk_fname = "labels_{:01d}_3T_T1w_MPR1_img".format(number)

    def fnames(self):
        lr_fname = path.join(self.path, 'LR', self.img_fname + ".nii.gz")
        hr_fname = path.join(self.path, 'HR', self.img_fname + ".nii.gz")
        msk_fname = path.join(self.path, 'MSK', self.msk_fname + ".nii.gz")
        return lr_fname, hr_fname, msk_fname

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
