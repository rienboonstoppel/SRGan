import nibabel as nib
import numpy as np
import os
from tensorboard.backend.event_processing import event_accumulator
import cv2
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import normalized_root_mse as NRMSE
import torch
import torchio as tio
import wandb

def print_config(config, args):
    print('Starting a run with config:')

    print_args = ['root_dir', 'num_workers',
                  'gpus', 'max_epochs', 'precision', 'warmup_batches',  # 'max_time'
                  'std', 'middle_slices', 'every_other', 'sampler']
    print("{:<20}| {:<10}".format('Var', 'Value'))
    print('-' * 22)
    for key in config:
        # if key != 'patients_dist' and key != 'patients_frac':
        print("{:<20}| {:<10} ".format(key, config[key]))

    for arg in print_args:
        print("{:<20}| {:<10} ".format(arg, getattr(args, arg)))

    if args.gan:
        print("{:<20}| {:<10} ".format('GAN', 'True'))
    else:
        print("{:<20}| {:<10} ".format('GAN', 'False'))

    if args.name:
        print("{:<20}| {:<10} ".format('name', args.name))
    else:
        print("{:<20}| {:<10} ".format('name', wandb.run.name))


    if args.no_checkpointing:
        print("{:<20}| {:<10} ".format('checkpointing', 'best only'))
    else:
        print("{:<20}| {:<10} ".format('checkpointing', 'best and time'))


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
    img = img.numpy()[0]
    img *= max_val
    img_nifti = nib.Nifti1Image(img, affine=affine, header=header)
    nib.save(img_nifti, fname)


def save_subject(subject, header, pref, max_vals, source, path='output'):
    os.makedirs(path, exist_ok=True)
    save_to_nifti(img=subject['LR'],
                  header=header,
                  max_val=max_vals['LR'],
                  fname=os.path.join(path, '{}_LR.nii.gz'.format(pref)),
                  source=source,
                  )

    save_to_nifti(img=subject['SR'],
                  header=header,
                  max_val=max_vals['SR'],
                  fname=os.path.join(path, '{}_SR.nii.gz'.format(pref)),
                  source=source,
                  )
    if source == 'sim' or source == 'hcp':# or source == 'mrbrains18':
        save_to_nifti(img=subject['HR'],
                      header=header,
                      max_val=max_vals['HR'],
                      fname=os.path.join(path, '{}_HR.nii.gz'.format(pref)),
                      source=source,
                      )


def save_subject_real(subject, header, pref, std, max_vals, source, path='output'):
    save_to_nifti(img=subject['LR'],
                  header=header,
                  std=std,
                  max_val=max_vals[0],
                  fname=os.path.join(path, '{}_LR.nii.gz'.format(pref)),
                  source=source,
                  )
    save_to_nifti(img=subject['GT'],
                  header=header,
                  std=std,
                  max_val=max_vals[1],
                  fname=os.path.join(path, '{}_GT.nii.gz'.format(pref)),
                  source=source,
                  )
    save_to_nifti(img=subject['SR'],
                  header=header,
                  std=std,
                  max_val=max_vals[2],
                  fname=os.path.join(path, '{}_SR.nii.gz'.format(pref)),
                  source=source,
                  )


def save_subject_all(subject, header, pref, std, max_vals, source, path='output'):
    for key in subject.keys():
        print(key)
        os.makedirs(path, exist_ok=True)
        save_to_nifti(img=subject[key],
                      header=header,
                      std=std,
                      max_val=max_vals[key[:2]],
                      fname=os.path.join(path, '{}_{}.nii.gz'.format(pref, key)),
                      source=source,
                      )


def save_images_from_event(path):
    event_acc = event_accumulator.EventAccumulator(path, size_guidance={'images': 0})
    event_acc.Reload()
    path, fname = os.path.split(path)
    for tag in event_acc.Tags()['images']:
        events = event_acc.Images(tag)
        tag_name = tag.replace('/', ' ')
        tag_path = os.path.join(path, 'images', tag_name)
        os.makedirs(tag_path, exist_ok=True)
        for index, event in enumerate(events):
            s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
            image = cv2.imdecode(s, cv2.IMREAD_GRAYSCALE)
            fname = '{:04}.jpg'.format(index)
            cv2.imwrite(os.path.join(tag_path, fname), image)


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
    metrics = ['NCC', 'SSIM', 'NRMSE']
    scores = {key: [] for key in metrics}
    # HR_aggs = []
    SR_aggs = []
    for i in range(len(output_data)):
        for j in range(len(output_data[i])):
            imgs_hr, imgs_sr, locations = output_data[i][j]
            HR_aggregator.add_batch(imgs_hr.unsqueeze(4), locations)
            SR_aggregator.add_batch(imgs_sr.unsqueeze(4), locations)

        HR_agg = HR_aggregator.get_output_tensor()
        SR_agg = SR_aggregator.get_output_tensor()
        # HR_aggs.append(HR_agg*self.args.std)
        SR_aggs.append(SR_agg * std)
        bg_idx, brain_idx = post_proc_info[i]
        HR_agg = post_proc(HR_agg, bg_idx, brain_idx) * std
        SR_agg = post_proc(SR_agg, bg_idx, brain_idx) * std

        HR_agg = HR_agg - np.mean(HR_agg)
        SR_agg = SR_agg - np.mean(SR_agg)
        scores['SSIM'].append(SSIM(HR_agg, SR_agg, gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
        scores['NCC'].append(NCC(HR_agg, SR_agg))
        scores['NRMSE'].append(NRMSE(HR_agg, SR_agg))

    return SR_aggs, {
        'SSIM': {
            'mean': np.mean(scores['SSIM']),
            'quartiles': np.percentile(scores['SSIM'], [25, 50, 75]),
        },
        'NCC': {
            'mean': np.mean(scores['NCC']),
            'quartiles': np.percentile(scores['NCC'], [25, 50, 75]),
        },
        'NRMSE': {
            'mean': np.mean(scores['NRMSE']),
            'quartiles': np.percentile(scores['NRMSE'], [25, 50, 75]),
        }
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

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def square_mask(slice, size):
    mask = np.zeros_like(slice)
    if isinstance(size, int):
        mask[int(slice.shape[0]/2-size/2):int(slice.shape[0]/2+size/2),
             int(slice.shape[0]/2-size/2):int(slice.shape[0]/2+size/2)] = 1
    if isinstance(size, tuple):
        if size[1]<=size[0]:
            return 'outer size must be larger dan inner size'
        mask[int(slice.shape[0]/2-size[1]/2):int(slice.shape[0]/2+size[1]/2),
             int(slice.shape[0]/2-size[1]/2):int(slice.shape[0]/2+size[1]/2)] = 1
        mask[int(slice.shape[0]/2-size[0]/2):int(slice.shape[0]/2+size[0]/2),
             int(slice.shape[0]/2-size[0]/2):int(slice.shape[0]/2+size[0]/2)] = 0
    return mask


def cuboid_mask(img3d, size):
    if type(img3d) == np.ndarray:
        mask = np.zeros_like(img3d)
    else:
        mask = torch.zeros_like(img3d)
    if isinstance(size, int):
        mask[int(img3d.shape[0]/2-size/2):int(img3d.shape[0]/2+size/2),
             int(img3d.shape[0]/2-size/2):int(img3d.shape[0]/2+size/2), :] = 1
    if isinstance(size, tuple):
        if size[1]<=size[0]:
            return 'outer size must be larger dan inner size'
        mask[int(img3d.shape[0]/2-size[1]/2):int(img3d.shape[0]/2+size[1]/2),
             int(img3d.shape[0]/2-size[1]/2):int(img3d.shape[0]/2+size[1]/2), :] = 1
        mask[int(img3d.shape[0]/2-size[0]/2):int(img3d.shape[0]/2+size[0]/2),
             int(img3d.shape[0]/2-size[0]/2):int(img3d.shape[0]/2+size[0]/2), :] = 0
    return mask
