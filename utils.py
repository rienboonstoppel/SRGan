import nibabel as nib
import numpy as np
import os

def print_config(config, args):
    print_args =['std', 'num_workers', 'root_dir', 'name', 'precision', 'gpus', 'max_epochs', 'max_time'] #'warmup_batches'

    print("{:<15}| {:<10}".format('Var', 'Value'))
    print('-'*22)
    for key in config:
        print("{:<15}| {:<10} ".format(key, config[key]))

    for arg in print_args:
        print("{:<15}| {:<10} ".format(arg, getattr(args, arg)))

    # if args.gan:
    #     print("{:<15}| {:<10} ".format('GAN', 'relativistic average'))
    # else:
    #     print("{:<15}| {:<10} ".format('GAN', 'vanilla'))

def re_scale(img, std, max_val):
    img *= std
    img *= max_val
    return img

def save_to_nifti(img, header, fname, std, max_val, source):
    affine = np.eye(4)
    if source == 'sim':
        affine[2,2] = 2
    img = img.numpy()[0]
    img = re_scale(img, std, max_val)
    img_nifti = nib.Nifti1Image(img, affine=affine, header=header)
    nib.save(img_nifti, fname)

def save_subject(subject, header, pref, std, max_vals, source, path='output'):
    save_to_nifti(img = subject['LR'],
                  header = header,
                  std = std,
                  max_val = max_vals[0],
                  fname = os.path.join(path, '{}_LR.nii.gz'.format(pref)),
                  source = source,
                  )
    save_to_nifti(img = subject['HR'],
                  header = header,
                  std = std,
                  max_val = max_vals[1],
                  fname = os.path.join(path, '{}_HR.nii.gz'.format(pref)),
                  source = source,
                  )
    save_to_nifti(img = subject['SR'],
                  header = header,
                  std = std,
                  max_val = max_vals[2],
                  fname = os.path.join(path, '{}_SR.nii.gz'.format(pref)),
                  source = source,
                  )

def save_subject_real(subject, header, pref, std, max_vals, source, path='output'):
    save_to_nifti(img = subject['LR'],
                  header = header,
                  std = std,
                  max_val = max_vals[0],
                  fname = os.path.join(path, '{}_LR.nii.gz'.format(pref)),
                  source = source,
                  )
    save_to_nifti(img = subject['GT'],
                  header = header,
                  std = std,
                  max_val = max_vals[1],
                  fname = os.path.join(path, '{}_GT.nii.gz'.format(pref)),
                  source = source,
                  )
    save_to_nifti(img = subject['SR'],
                  header = header,
                  std = std,
                  max_val = max_vals[2],
                  fname = os.path.join(path, '{}_SR.nii.gz'.format(pref)),
                  source = source,
                  )

def save_subject_all(subject, header, pref, std, max_vals, source, path='output'):
    for key in subject.keys():
        print(key)
        save_to_nifti(img = subject[key],
                      header = header,
                      std = std,
                      max_val = max_vals[key[:2]],
                      fname = os.path.join(path, '{}_{}.nii.gz'.format(pref, key)),
                      source = source,
                      )