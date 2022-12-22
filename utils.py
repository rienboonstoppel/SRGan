import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity as SSIM
import torch
import torchio as tio
from skimage import filters


def save_to_nifti(img, header, fname, max_val, source):
    if source == 'sim':
        affine = np.array([[-0.7, 0, 0, 0],
                           [0, -0.7, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    elif source == 'mrbrains':
        affine = np.array([[0.9583, 0, 0, 0],
                           [0, 0.9583, 0, 0],
                           [0, 0, 3, 0],
                           [0, 0, 0, 1]])
    elif source == 'hcp':
        affine = np.array([[0.7, 0, 0, 0],
                           [0, 0.7, 0, 0],
                           [0, 0, 0.7, 0],
                           [0, 0, 0, 1]])
    elif source == 'oasis':
        affine = np.array([[-1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    img *= max_val
    img_nifti = nib.Nifti1Image(img, affine=affine, header=header)
    nib.save(img_nifti, fname)


def NCC(real_image, generated_image):
    """Method to compute the normalised cross correlation between two images.
    Arguments:
                real_image:       (numpy array) the real image
                predicted_image:  (numpy array) the predicted image by the model
    Returns:
                NCCScore:         (float) the normalised cross correlation score
    """
    # if the images are not the same size, raise an error
    if real_image.shape != generated_image.shape:
        raise AssertionError("The inputs must be the same size.")
    # reshape images to vectors
    u = real_image.reshape((real_image.shape[0] * real_image.shape[1] * real_image.shape[2], 1))
    v = generated_image.reshape((generated_image.shape[0] * generated_image.shape[1] * real_image.shape[2], 1))
    # take the real image and subtract the mean of the real image
    u = u - u.mean(keepdims=True)
    # take the generated image and subtract the mean of the generated image
    v = v - v.mean(keepdims=True)
    # transpose the real image for multiplication
    TransposedU = np.transpose(u)
    # calculate the length of the image
    length = np.linalg.norm(u, ord=2) * np.linalg.norm(v, ord=2)
    # calculate the NCC of the real image and the generated image
    NCCScore = float(TransposedU.dot(v)) / length
    # return the NCC score
    return NCCScore


def post_proc(img: torch.Tensor, bg_idx: np.ndarray, crop_coords: tuple) -> np.ndarray:
    img[bg_idx] = 0
    min, max = crop_coords
    img = img.squeeze(0)[min[0]:max[0] + 1, min[1]:max[1] + 1, min[2]:max[2] + 1].numpy()
    return img


def imgs_cat(imgs_lr, imgs_hr, imgs_sr):
    imgs_lr = imgs_lr[:10].squeeze()
    imgs_hr = imgs_hr[:10].squeeze()
    imgs_sr = imgs_sr[:10].squeeze()
    diff = (imgs_hr - imgs_sr) * 2 + .5
    img_grid = torch.cat([imgs_lr, imgs_hr, imgs_sr, diff], dim=0).unsqueeze(1)
    return img_grid


def val_metrics(output_data, HR_aggregator, SR_aggregator, std, post_proc_info):
    metrics = ['SSIM', 'NCC']
    scores = {key: [] for key in metrics}
    SR_aggs = []
    for i in range(len(output_data)):
        for j in range(len(output_data[i])):
            imgs_hr, imgs_sr, locations = output_data[i][j]
            HR_aggregator.add_batch(imgs_hr.unsqueeze(4), locations)
            SR_aggregator.add_batch(imgs_sr.unsqueeze(4), locations)

        HR_agg = HR_aggregator.get_output_tensor()
        SR_agg = SR_aggregator.get_output_tensor()
        SR_aggs.append(SR_agg * std)
        bg_idx, brain_idx = post_proc_info[i]
        HR_agg = post_proc(HR_agg, bg_idx, brain_idx) * std
        SR_agg = post_proc(SR_agg, bg_idx, brain_idx) * std

        HR_agg = HR_agg - np.mean(HR_agg)
        SR_agg = SR_agg - np.mean(SR_agg)
        scores['SSIM'].append(
            SSIM(HR_agg, SR_agg, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.5))
        scores['NCC'].append(NCC(HR_agg, SR_agg))

    return SR_aggs, {
        'SSIM': {
            'mean': np.mean(scores['SSIM']),
            'quartiles': np.percentile(scores['SSIM'], [25, 50, 75]),
        },
        'NCC': {
            'mean': np.mean(scores['NCC']),
            'quartiles': np.percentile(scores['NCC'], [25, 50, 75]),
        },
    }


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for imgs in dataloader:
        data = imgs['HR'][tio.DATA].squeeze(4)
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std
