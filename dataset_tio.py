import os
from abc import abstractmethod
from os import path
import numpy as np
import nibabel as nib
from glob import glob
import random
import torch
import torchio as tio
from torchio.data.subject import Subject
import cv2
from skimage import filters
from tqdm import tqdm

def perc_norm(img3d, perc=95):
    """
    Normalize 3d image based on its 95 percentile value
    """
    img3d = np.clip(img3d - img3d[0, 0, 0], 0, None)
    max_val = np.percentile(img3d, perc)
    img_norm = img3d.astype(float) / max_val.astype(np.float32)
    return img_norm, max_val


def select_slices(img, middle_slices, every_other=1):
    """
    Select slices from the middle of a volume, optionally skipping slices based on param every_other
    """
    diff = (img.shape[2] - middle_slices) / 2
    img = img[:, :, int(np.ceil(diff)):img.shape[2] - int(np.floor(diff)):every_other]
    return img


def augment_usm(img3d, amount=0.5):
    """
    Apply unsharp masking to a volume, sharpening only in plane slices
    """
    img3d_aug = img3d + amount * (img3d - filters.gaussian(img3d, sigma=(1, 1, 0), preserve_range=True))
    return img3d_aug


class Image(object):
    """
    Base class for different datasets
    """

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
            imgs_np['LR'] = augment_usm(imgs_np['LR'])

        if 'HR' in imgs_np.keys():
            imgs_np['HR'], self.scaling_HR = perc_norm(imgs_np['HR'])

        subject = tio.Subject({key: tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims((imgs_np[key]), 0)))
                               for key in imgs_np.keys() if key != 'MSK'})

        if 'MSK' in imgs_np.keys():
            if ((imgs_np['MSK'] == 0) | (imgs_np['MSK'] == 1)).all():  # if mask is binary
                msk_eroded = cv2.erode(imgs_np['MSK'], np.ones((10, 10)), iterations=3)
                subject.add_image(tio.LabelMap(tensor=torch.from_numpy(np.expand_dims(msk_eroded, 0))),
                                  'MSK_eroded')  # used for label-sampler
                subject.add_image(tio.LabelMap(tensor=torch.from_numpy(np.expand_dims(imgs_np['MSK'], 0))),
                                  'MSK')  # used for metric calculation
            else:
                imgs_np['MSK'][imgs_np['MSK'] > 0] = 1  # if mask has more class labels (background is assumed label 0)
                msk_eroded = cv2.erode(imgs_np['MSK'], np.ones((10, 10)), iterations=3)
                subject.add_image(tio.LabelMap(tensor=torch.from_numpy(np.expand_dims(msk_eroded, 0))), 'MSK_eroded')
                subject.add_image(tio.LabelMap(tensor=torch.from_numpy(np.expand_dims(imgs_np['MSK'], 0))), 'MSK')
        return subject

    def info(self) -> dict:
        """
        Keep some info on the original images to save SR images in the proper format when generated
        """
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
    """
    Subclass for simulated images
    """

    def __init__(self, number, root_dir='data', middle_slices=50, every_other=1,
                 augment=False):
        super().__init__(middle_slices, every_other, augment)
        self.path = os.path.join(root_dir, "brain_simulated_t1w_mri")
        self.img_fname = "08-Apr-2022_Ernst_labels_{:06d}_" \
                         "3T_T1w_MPR1_img_act_1_contrast_1".format(number)

    def fnames(self) -> dict:
        lr_fname = path.join(self.path, 'LR_' + 'img', self.img_fname + "_Res_1_1_1_" + 'img' + ".nii.gz")
        hr_fname = path.join(self.path, 'HR_' + 'img', self.img_fname + "_Res_0.7_0.7_1_" + 'img' + ".nii.gz")
        msk_fname = path.join(self.path, 'HR_' + 'msk', self.img_fname + "_Res_0.7_0.7_1_" + 'msk' + ".nii.gz")

        return {'LR': lr_fname,
                'HR': hr_fname,
                'MSK': msk_fname}


class MRBrainS18Image(Image):
    """
    Subclass for MRBrainS18 images
    """

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
    """
    Subclass for HCP images
    """

    def __init__(self, number, root_dir='data', middle_slices=50, every_other=1, augment=False):
        super().__init__(middle_slices, every_other, augment)
        self.path = os.path.join(root_dir, 'brain_real_t1w_mri', 'HCP')
        self.img_fname = "{:06d}_3T_T1w_MPR1_img".format(number)
        self.msk_fname = "labels_{:06d}_3T_T1w_MPR1_img".format(number)

    def fnames(self) -> dict:
        lr_fname = path.join(self.path, 'LR', self.img_fname + ".nii.gz")
        hr_fname = path.join(self.path, 'HR', self.img_fname + ".nii.gz")
        msk_fname = path.join(self.path, 'MSK', self.msk_fname + ".nii.gz")
        return {'LR': lr_fname,
                'HR': hr_fname,
                'MSK': msk_fname}


class OASISImage(Image):
    """
    Subclass for OASIS images
    """

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
             nr_train_patients=30, nr_val_patients=10, nr_test_patients=10,
             middle_slices=50, every_other=1, root_dir='data', augment=False):
    """
    Generate list of simulated subjects
    """
    # define paths
    random.seed(21011998)
    path = os.path.join(root_dir, "brain_simulated_t1w_mri", 'HR_img/')
    fnames = glob(path + "*.nii.gz")
    ids = sorted(list(map(int, [(fnames[i][-64:-58]) for i in range(len(fnames))])))
    random.shuffle(ids)

    if nr_train_patients + nr_val_patients + nr_test_patients > 200:
        raise ValueError("Total number of patients should be 200 or less")

    if dataset == 'training':
        ids_split = ids[:nr_train_patients]
    elif dataset == 'validation':
        ids_split = ids[-nr_val_patients - nr_test_patients:-nr_test_patients]
    elif dataset == 'test':
        ids_split = ids[-nr_test_patients:]
    else:
        raise ValueError("Dataset '{}' not recognized, use 'training, 'validation' or 'test' instead".format(dataset))

    # make arrays
    subjects = []
    infos = []
    print('Loading simulated {} set...'.format(dataset))
    # for num in tqdm(ids_split, desc='Load {} set\t'.format(dataset), bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}',
    #                 leave=True, position=0):
    for num in ids_split:
        data = SimImage(num, root_dir, middle_slices, every_other, augment=augment)
        subjects.append(data.subject())
        info = data.info()
        info['id'] = num
        infos.append(info)
    return subjects, infos


def HCP_data(dataset,
             nr_train_patients=30, nr_val_patients=10, nr_test_patients=10,
             middle_slices=50, every_other=1, root_dir='data', augment=False):
    """
    Generate list of HCP subjects
    """
    random.seed(21011998)
    path = root_dir + "/brain_real_t1w_mri/HCP/HR/"
    fnames = glob(path + "*.nii.gz")
    ids = sorted(list(map(int, [(fnames[i][-29:-23]) for i in range(len(fnames))])))
    random.shuffle(ids)

    if nr_train_patients + nr_val_patients + nr_test_patients > 50:
        raise ValueError("Total number of patients should be 50 or less")

    if dataset == 'training':
        ids_split = ids[:nr_train_patients]
    elif dataset == 'validation':
        ids_split = ids[-nr_val_patients - nr_test_patients:-nr_test_patients]
    elif dataset == 'test':
        ids_split = ids[-nr_test_patients:]
    else:
        raise ValueError("Dataset '{}' not recognized, use 'training, 'validation' or 'test' instead".format(dataset))

    # make arrays
    subjects = []
    infos = []
    print('Loading HCP {} set...'.format(dataset))

    for num in ids_split:
        data = HCPImage(num, root_dir=root_dir, middle_slices=middle_slices, every_other=every_other, augment=augment)
        subjects.append(data.subject())
        info = data.info()
        info['id'] = num
        infos.append(info)
    return subjects, infos


def MRBrainS18_data(dataset,
                    root_dir='data', middle_slices=50, every_other=1, augment=False):
    """
    Generate list of MRBrainS18 subjects
    """
    path = root_dir + "/brain_real_t1w_mri/MRBrainS18/GT/"
    fnames = glob(path + "*.nii.gz")
    ids = sorted(list(map(int, [(fnames[i][-15:-14]) for i in range(len(fnames))])))

    if dataset == 'validation':
        ids_split = ids[:3]
    elif dataset == 'test':
        ids_split = ids[3:]
    else:
        raise ValueError("Dataset '{}' not recognized, use 'validation' or 'test' instead".format(dataset))

    # make arrays
    subjects = []
    infos = []
    print('Loading MRBrainS18 dataset...')
    for num in ids_split:
        data = MRBrainS18Image(num, root_dir=root_dir, middle_slices=middle_slices, every_other=every_other,
                               augment=augment)
        subjects.append(data.subject())
        info = data.info()
        info['id'] = num
        infos.append(info)
    return subjects, infos


def OASIS_data(dataset,
               root_dir='data', middle_slices=50, every_other=1, augment=False):
    """
    Generate list of OASIS subjects
    """
    path = root_dir + "/brain_real_t1w_mri/OASIS/LR/"
    fnames = glob(path + "*.nii.gz")
    ids = sorted(list(map(int, [(fnames[i][-46:-42]) for i in range(len(fnames))])))

    if dataset == 'validation':
        ids_split = ids[:5]
    elif dataset == 'test':
        ids_split = ids[5:]
    else:
        raise ValueError("Dataset '{}' not recognized, use 'validation' or 'test' instead".format(dataset))

    # make arrays
    subjects = []
    infos = []
    print('Loading OASIS dataset...')
    for num in ids_split:
        data = OASISImage(num, root_dir=root_dir, middle_slices=middle_slices, every_other=every_other, augment=augment)
        subjects.append(data.subject())
        info = data.info()
        info['id'] = num
        infos.append(info)
    return subjects, infos


def data(dataset, nr_hcp_train=30, nr_sim_train=30, nr_hcp_val=10, nr_sim_val=10,
         middle_slices=None, root_dir='data', every_other=1):
    """
    Generate mixed list of Sim and HCP subjects
    """
    subjects = []
    if dataset == 'training':
        if nr_hcp_train != 0:
            hcp_subjects, _ = HCP_data(dataset='training', nr_train_patients=nr_hcp_train, middle_slices=middle_slices,
                                       root_dir=root_dir, every_other=every_other)
            subjects.extend(hcp_subjects)
            print('Loaded HCP {} dataset with length {}'.format(dataset, len(hcp_subjects)))
        if nr_sim_train != 0:
            sim_subjects, _ = sim_data(dataset='training', nr_train_patients=nr_sim_train, middle_slices=middle_slices,
                                       root_dir=root_dir, every_other=every_other)
            subjects.extend(sim_subjects)
            print('Loaded simulated {} dataset with length {}'.format(dataset, len(sim_subjects)))
    if dataset == 'validation':
        if nr_hcp_val != 0:
            hcp_subjects, _ = HCP_data(dataset='validation', nr_train_patients=nr_hcp_train, nr_val_patients=nr_hcp_val, middle_slices=middle_slices,
                                       root_dir=root_dir, every_other=every_other)
            subjects.extend(hcp_subjects)
            print('Loaded HCP {} dataset with length {}'.format(dataset, len(hcp_subjects)))
        if nr_sim_val != 0:
            sim_subjects, _ = sim_data(dataset='validation', nr_train_patients=nr_hcp_train, nr_val_patients=nr_sim_val, middle_slices=middle_slices,
                                       root_dir=root_dir, every_other=every_other)
            subjects.extend(sim_subjects)
            print('Loaded simulated {} dataset with length {}'.format(dataset, len(sim_subjects)))

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
