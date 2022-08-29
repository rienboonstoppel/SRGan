import numpy as np
import nibabel as nib
from glob import glob
import os
from scipy import signal
from tqdm import tqdm

def create_2d_fft_mask(HR_slice: np.array, downscale_factor = 0.7, alpha=0.5) -> np.array:
    padding = int((HR_slice.shape[0] - (1 / (1 - .5 * alpha)) * HR_slice.shape[0] * downscale_factor) / 2)
    x = signal.tukey(HR_slice.shape[0] - 2 * padding, alpha)
    y = signal.tukey(HR_slice.shape[0] - 2 * padding, alpha)
    [mask_x, mask_y] = np.meshgrid(x, y)
    mask = mask_x * mask_y
    mask_padded = np.pad(mask, padding, constant_values=0)
    return mask_padded

root_dir = 'data/brain_real_t1w_mri'
data_source = 'HCP'
data_resolution = 'HR'
path = os.path.join(root_dir, data_source, data_resolution)
fnames = glob(path + "/*.nii.gz")

downscale_factor = 0.7
alpha = 0.5

save_folder = 'LR'
os.makedirs(os.path.join(root_dir, data_source, save_folder), exist_ok=True)

mask_ft = create_2d_fft_mask(HR_slice=nib.load(fnames[0]).get_fdata()[:,:,0], downscale_factor=downscale_factor, alpha=alpha)

for i in tqdm(range(len(fnames))):
    img = nib.load(fnames[i])
    HR = img.get_fdata()
    LR = np.zeros_like(HR)
    for j in range(HR.shape[2]):
        kspace = np.fft.fftshift(np.fft.fft2(HR[:, :, j]))
        kspace_filtered = mask_ft * kspace
        LR[:, :, j] = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_filtered)))
    LR_img = nib.Nifti1Image(LR, img.affine, img.header)
    save_path = os.path.join(root_dir, data_source, save_folder)
    save_name = os.path.split(fnames[i])[1]#[:-7] + '_downscaled.nii.gz'
    nib.save(LR_img, os.path.join(save_path, save_name))
