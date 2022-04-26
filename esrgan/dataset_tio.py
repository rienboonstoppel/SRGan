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


def perc_norm(img3d, perc=95):
    max_val = np.percentile(img3d, perc)
    img_norm = img3d.astype(float)  / max_val.astype(np.float32)
    return img_norm, max_val


def slice_middle(img, numslices):
    diff = (img.shape[2]-numslices)/2
    img = img[:, :, int(np.ceil(diff)):img.shape[2]-int(np.floor(diff))]
    return img


class ImagePair(object):
    def __init__(self, number, root_dir='data', select_slices=50, simulated=True):
        self._number = number
        self.select_slices = select_slices
        self.simulated = simulated
        if simulated:
            self.path = root_dir + "/brain_simulated_t1w_mri"
            self.img_fname = "23-Aug-2021_Ernst_labels_{:06d}_" \
                             "3T_T1w_MPR1_img_act_1_contrast_1".format(self._number)
        else:
            self.path = root_dir + "/brain_real_t1w_mri"
            self.img_fname = "p{:01d}_reg_T1".format(self._number)

    def LR_HR_fnames(self):
        if self.simulated:
            LRf = path.join(self.path, 'LR', self.img_fname + "_Res_2_2_2_img.nii.gz")
            HRf = path.join(self.path, 'HR', self.img_fname + "_Res_1_1_2_img.nii.gz")
        else:
            LRf = path.join(self.path, 'LR', self.img_fname + ".nii.gz")
            HRf = path.join(self.path, 'GT', self.img_fname + ".nii.gz")
        return LRf, HRf

    def to_nifty(self):
        LR_fname, HR_fname = self.LR_HR_fnames()
        self.LR = nib.load(LR_fname)
        self.HR = nib.load(HR_fname)

    def subject(self):
        self.to_nifty()

        if self.select_slices is not None:
            LR = slice_middle(self.LR.get_fdata(), self.select_slices)
            HR = slice_middle(self.HR.get_fdata(), self.select_slices)
        else:
            LR = self.LR.get_fdata()
            HR = self.LR.get_fdata()

        LR_norm, self.scaling_LR = perc_norm(LR)
        HR_norm, self.scaling_HR = perc_norm(HR)

        subject = tio.Subject(
            LR=tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims(LR_norm,0))),
            HR=tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims(HR_norm,0))),
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


def data_split(dataset, patients_frac=1, train_frac=0.7, val_frac=.15, test_frac=.15, numslices=50, root_dir='data',
               randomseed=21011998, simulated=True):
    # define paths
    random.seed(randomseed)
    path   = root_dir + "/brain_simulated_t1w_mri/HR/"
    fnames = glob(path + "*.nii.gz")
    ids    = sorted(list(map(int, [(fnames[i][-60:-54]) for i in range(len(fnames))])))
    random.shuffle(ids)

    # define data splits
    split1 = int(np.floor(patients_frac*train_frac*len(ids)))
    split2 = split1+int(np.floor(patients_frac*val_frac*len(ids)))
    split3 = split2+int(np.floor(patients_frac*test_frac*len(ids)))

    if dataset == 'training':
        ids_split = ids[:split1]
    elif dataset == 'validation':
        ids_split = ids[split1:split2]
    elif dataset == 'test':
        ids_split = ids[split2:split3]

    # make arrays
    subjects = []
    print('Loading {} set...'.format(dataset))
    # for num in tqdm(ids_split, desc='Load {} set\t'.format(dataset), bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}',
    #                 leave=True, position=0):
    for num in ids_split:
        data = ImagePair(num, root_dir=root_dir, select_slices=numslices, simulated=simulated)
        subjects.append(data.subject())
    return subjects


def real_data(numslices=45, root_dir='data', simulated=False):
    path   = root_dir + "/brain_real_t1w_mri/GT/"
    fnames = glob(path + "*.nii.gz")
    ids    = sorted(list(map(int, [(fnames[i][-15:-14]) for i in range(len(fnames))])))

    # make arrays
    subjects = []
    print('Loading real dataset...')

    for num in ids:
        data = ImagePair(num, root_dir=root_dir, select_slices=numslices, simulated=simulated)
        subjects.append(data.subject())
    return subjects

# TODO fix only even overlap
def calculate_overlap(img, patch_size, ovl_perc):
    patch_size = np.array([patch_size[0], patch_size[1]])
    ovl_perc = np.array([ovl_perc[0], ovl_perc[1]])
    size = img.shape
    sizeXY = np.array([size[1], size[2]])
    nr_patches = np.divide(sizeXY, ovl_perc * patch_size).astype(int) - 1
    residual = sizeXY - (patch_size + (nr_patches - 1) * ovl_perc * patch_size)
    overlap = (patch_size * np.array(ovl_perc) + np.ceil(np.divide(residual, nr_patches))).astype(int)
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
