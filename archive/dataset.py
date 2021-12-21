from os import path
import numpy as np
import nibabel as nib
from tqdm import tqdm
from torch.utils.data import Dataset
from glob import glob
import random
import torch


class ImagePair(object):
    def __init__(self, number, root_dir='data'):
        self._number = number
        self.path = root_dir + "/brain_simulated_t1w_mri"
        self.img_fname = "23-Aug-2021_Ernst_labels_{:06d}_" \
                         "3T_T1w_MPR1_img_act_1_contrast_1".format(self._number)

    def LR_HR_fnames(self):
        LR_suff = "_Res_2_2_2_img.nii.gz"
        LRf = path.join(self.path, 'LR', self.img_fname + LR_suff)
        HR_suff = "_Res_1_1_2_img.nii.gz"
        HRf = path.join(self.path, 'HR', self.img_fname + HR_suff)
        return LRf, HRf

    def img(self):
        LR_fname, HR_fname = self.LR_HR_fnames()
        self.LR = nib.load(LR_fname)
        self.HR = nib.load(HR_fname)
        img_pair = {
            'LR': self.LR.get_fdata(),
            'HR': self.HR.get_fdata()
        }
        return img_pair

    def info(self):
        img_info = {
            'LR': self.LR.header,
            'HR': self.HR.header
        }
        return img_info


def data_split(dataset, patients_frac=1, train_frac=0.7, val_frac=.15, test_frac=.15, numslices=50, root_dir='data',
               randomseed=7886462529168327085):
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
    samples = list()
    for num in tqdm(ids_split, desc='Load {} set\t'.format(dataset), bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}'):
        data = ImagePair(num, root_dir=root_dir)
        img_pair = data.img()
        LR, start_idx = slice_middle(img_pair['LR'], numslices=numslices)
        HR, _ = slice_middle(img_pair['HR'], numslices=numslices)
        for i in range(numslices):
            min_val, max_val = np.percentile(HR[:,:,i], (1, 99))
            LRi = ((LR[:,:,i].astype(float) - min_val) /
                   (max_val - min_val)).clip(0, 1).astype(np.float32)
            HRi = ((HR[:,:,i].astype(float) - min_val) /
                   (max_val - min_val)).clip(0, 1).astype(np.float32)
            slice_pair = {
                'LR': np.expand_dims(LRi, 0),
                'HR': np.expand_dims(HRi, 0)
            }
            sample = {
                'img': slice_pair,
                'id': num,
                'slice': i+start_idx
            }
            samples.append(sample)
    return samples


def slice_middle(img, numslices):
    diff = (img.shape[2]-numslices)/2
    img = img[:,:,int(np.ceil(diff)):img.shape[2]-int(np.floor(diff))]
    return img, int(np.ceil(diff))


class ImagePairDataset(Dataset):
    def __init__(self, dataset, patients_frac=1, train_frac=0.7, val_frac=.15, test_frac=.15, numslices=50, root_dir='data', transform=None):
        self._root_dir = root_dir
        self.transform = transform

        self._imgs = data_split(dataset,
                                patients_frac=patients_frac, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, numslices=numslices,
                                root_dir=root_dir,
                                )
        self._idcs = np.arange(len(self._imgs))

    def __len__(self):
        return len(self._idcs)

    def __getitem__(self, idx):

        img_idx = self._idcs[idx]
        img_pair = self._imgs[img_idx]

        sample = {'LR': img_pair['img']['LR'],
                  'HR': img_pair['img']['HR'],
                  'id': 'Im_{}-Slc_{}'.format(img_pair['id'], img_pair['slice']),
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.fixed_conversion = torch.from_numpy

    def __call__(self, sample: dict) -> dict:
        LR = sample['LR']
        HR = sample['HR']
        try:
            LR = self.fixed_conversion(LR)
            HR = self.fixed_conversion(HR)
        except ValueError:
            LR = self.fixed_conversion(np.ascontiguousarray(LR))
            HR = self.fixed_conversion(np.ascontiguousarray(HR))
        ret = {'LR': LR,
               'HR': HR,
               'id': sample['id']}
        return ret


class Normalize(torch.nn.Module):
    def __init__(self, mean=0, std=1, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        return {'LR': (sample['LR']-self.mean)/self.std,
                'HR': (sample['HR']-self.mean)/self.std,
                'id': sample['id']}


def MakePatchStartArray(image, patchSize = np.array([64,64]), ovlPerc = np.array([0.5, 0.5])):
    size03D  = (np.array(image.shape)).astype(np.int)
    # size03D  = [224, 224, 70]
    # show the X and Y sizes
    size0    = np.array([size03D[0], size03D[1]])
    # the third part of size03D shows the amount of slices in the image (Z direction)
    nrSlices = size03D[2]
    # define the amount of patches which need to be made, keep in mind the overlap of the patches (this is in X and in Y)
    numberOfPatches = (np.divide(size0, (ovlPerc * patchSize))).astype(int) - 1
    # define the image size which will be the output when setting all patches back to an image
    roundedFull = patchSize + (numberOfPatches - 1) * ovlPerc * patchSize
    # empty space that remains after dividing images in patches
    residual    = size0 - roundedFull
    # difference between each patch (in space) to divide the patches over the image equally
    deltaPix    = np.divide(residual, (numberOfPatches - 1))
    # start an empty list to add the start arrays of the X direction to
    startArrayY = []
    # start an empty list to add the start arrays of the Y direction to
    startArrayX = []
    # loop over the amount of patches needed in the X direction
    for x in range(numberOfPatches[0]):
        # define the startpoint for every slice in X direction
        startArrayX.append(int(ovlPerc[0] * patchSize[0] * x + deltaPix[0] * x + 0.5))
    # loop over the amount of patches needed in the Y direction
    for y in range(numberOfPatches[1]):
        # define the startpoint for every slice in Y direction
        startArrayY.append(int(ovlPerc[1] * patchSize[1] * y + deltaPix[1] * y + 0.5))
    # return the start position of X and Y, the amount of patches in the slice and the
    # number of slices in the image when the method is called
    return startArrayX,startArrayY,numberOfPatches,nrSlices,size03D,size0