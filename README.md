Very short and compact README.

**Main components of code:**

1) `Main.py`
Run for training. Uses couple ways to get all the parameters
- `Argparser` in `main`: (train-specific arguments)
  - `num_workers`: (int) number of cpu cores available
  - `root_dir`: (str) root directory of main.py
  - `name`: (str) name of your training run
  - `warmup_batches`: (int) amount of batches the training will start with only pixel-loss
  - `gan`: if flag present, training will run as adversarial loss, else not
  - `no-checkpointing`: if flag present, only the best checkpoint (based on SSIM) will be saved, else every epoch
- `Argparser` in `trainer` (data-specific arguments)
  - `std`: (float) std of data, for normalisation
  - `middle_slices`: (int) number of slices to select from the middle of the volume of a subject
  - `every_other`: (int) select every x other slices from subject 
  - `sampler`: (str) grid of label-sampler, for labelsampler, mask is required
- `Argparser` builtin Pytorch Lighting 
  - `gppus`: (int) number of gpus available
  - `max_epochs`: (int) set max amount of epochs
- `config` in `main` (dict containing all possible vars that can be varied)
  - `optimizer`: (str) adam or sgd,
  - `b1`, `b2`: (floats) betas for adam (if needed)
  - `batch_size`: (int) batch size for training
  - `num_filters`: (int) number of filters in architecture
  - `learning_rate_G`: (float) learning rate of generator,
  - `learning_rate_D`: (float) learning rate of discriminator (if needed),
  - `patch_size`: (int) patch size ,
  - `alpha_edge`: (float) weighting of edge-loss,
  - `alpha_pixel`: (float) weighting of pixel-loss,
  - `alpha_perceptual`: (float) weighting of perceptual (VGG) loss,
  - `alpha_adversarial`: (float) weighting of adversarial loss (if needed),
  - `ragan`: (bool) vanilla gan of relativistic average gan (if needed)
  - `gan_mode`: (str) kind of adversarial loss (vanilla, lsgan, wgan) (if needed)
  - `edge_loss`: (int) kind of edge losse (1, 2, 3, see `edgeloss.py`)
  - `netD_freq`: (int) frequency the discriminator is trained vs the generator (if needed),
  - `datasource`: (str) type of dataset, (1mm_07mm or 2mm-1mm)
  - `patients_frac`: (float 0-1) percentage of data that will be used for training,
  - `patch_overlap`: (float 0-1) percentage of overlap for patches, if gridsampler,
  - `generator`: generator architecture (ESRGAN, RRDB, FSRCNN, DeepUResnet)

2) `trainer_org.py` 
Accompanying trainer for non-GAN training. Written using Pytorch Lightning and logging to Wandb

3) `trainer_gan.py` 
Accompanying trainer for GAN training. Written using Pytorch Lightning and logging to Wandb

4) `sweep.yaml`
For a hparam sweep (everything in the config can be searched) using wandb, this file can be used and adapted 

5) `dataset_tio`
Baseclass for data, dataset is build using TorchIO, exploiting their patch-based pipeline

Data is not present in this repo, but should be located in the `data` folder in root
```
├── main.py
├── data                          # data-folder
│   ├── brain_real_t1w_mri        # real data
│   │   ├── GT                    # ground truth real data (1mm)
│   │   ├── LR                    # low-res real data (2mm)
│   │   └── MSK                   # segmentations of GT data
│   ├── brain_simulated_t1w_mri   # simulated data
│   │   ├── 1mm_07mm              # 1mm and 0.7mm simulated data
│   │   │   ├── HR_img            # high res simulated data (0.7mm)
│   │   │   ├── HR_msk            # segmentations of high res simulated data
│   │   │   └── LR_img            # low res simulated data (1mm)
│   │   └── 2mm_1mm               # 2mm and 1mm simulated data
│   │   │   ├── HR_img            # high res simulated data (1mm)
│   │   │   ├── HR_msk            # segmentations of high res simulated data 
│   │   │   └── LR_img            # low res simulated data (2mm) 
│   │   └── ... 
│   └── ... 
└── ...
```