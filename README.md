Very short and compact README.

**Main components of code:**

1) `Main.py`
Run for training. Uses a couple of methods to get all the parameters
- `Argparser` in `main`: (train-specific arguments)
  - `num_workers`: (int) number of cpu cores available
  - `root_dir`: (str) root directory of main.py
  - `name`: (str) name of your training run
  - `wandb_project`: (str) name of wandb project for logging
  - `warmup_batches`: (int) amount of batches the training will start with only pixel-loss
  - `gan`: if flag present, training will run with adversarial loss, else not
  - `no-checkpointing`: if flag present, only the best checkpoint (based on SSIM) will be saved, else every epoch
- `Argparser` in `trainer` (data-specific arguments)
  - `std`: (float) std of data, for normalisation
  - `middle_slices`: (int) number of slices to select from the middle of the volume of a subject
  - `every_other`: (int) select every x other slices from subject 
  - `sampler`: (str) grid of label-sampler, for labelsampler, mask is required
- `Argparser` builtin Pytorch Lighting 
  - `gpus`: (int) number of gpus available
  - `max_epochs`: (int) set max amount of epochs
- `config` in `main` (dict containing all possible vars that can be varied)
  - `optimizer`: (str) adam or sgd,
  - `b1`, `b2`: (floats) betas for adam (if needed)
  - `batch_size`: (int) batch size for training
  - `num_filters`: (int) number of filters in architecture of generator
  - `learning_rate_G`: (float) learning rate of generator,
  - `learning_rate_D`: (float) learning rate of discriminator (if needed),
  - `patch_size`: (int) patch size ,
  - `alpha_edge`: (float) weighting of edge-loss,
  - `alpha_pixel`: (float) weighting of pixel-loss,
  - `alpha_perceptual`: (float) weighting of perceptual (VGG) loss,
  - `alpha_adversarial`: (float) weighting of adversarial loss (if needed),
  - `gan_mode`: (str) kind of adversarial loss (vanilla (RaSGAN), wgan) (if needed)
  - `edge_loss`: (int) kind of edge losse (1, 2, 3, see `edgeloss.py`)
  - `netD_freq`: (int) frequency the discriminator is trained vs the generator (if needed),
  - `patch_overlap`: (float 0-1) percentage of overlap for patches, if gridsampler,
  - `generator`: generator architecture (ESRGAN, RRDB, FSRCNN, DeepUResnet, DeepUResnet_v2)
  - `nr_sim_train`: (int) number of simulated subjects for training
  - `nr_hcp_train`: (int) number of hcp subjects for training
  - `nr_sim_val`: (int) number of simulated subjects for validation
  - `nr_hcp_val`: (int) number of simulated subjects for validation

  It is meant to run on command line, for example run: `python main.py --gan --name mixed-data --wandb_project example --gpus 1 --log_every_n_steps 500 --max_epochs 25 --no_checkpointing`

2) `dataset_tio.py`
Baseclass for data, dataset is build using TorchIO, exploiting their patch-based pipeline

3) `trainer_org.py` 
Accompanying trainer for non-GAN training. Written using Pytorch Lightning and logging to Wandb

4) `trainer_gan.py` 
Accompanying trainer for GAN training. Written using Pytorch Lightning and logging to Wandb

5) `predict.py`
Run for SR generation using a model checkpoint.
To combine with USM: USM before SR, turn on augmentation in dataloading. <br />
USM after SR, uncomment couple of lines before saving SR. <br />
It is meant to run on command line, for example run: `python predict.py --gan --generator ESRGAN --source sim`

6) `calculate_scores.py`
Run to calculate metrics on generated SR images <br />
It is meant to run on command line, for example run: `python calculate_scores.py --source sim`

7) `sweep.yaml`
For a hparam sweep (everything in the config can be searched) using wandb

Data is not present in this repo, but should be located in the `data` folder in root
```
.
├── main.py
├── data                          # data-folder
│   ├── brain_real_t1w_mri        # real data
│   │   ├── MRBrainS18            # Data from MRBrainS18 challenge
│   │   │   ├── GT                # 1mm images
│   │   │   └── MSK               # segmentations of GT data
│   │   ├── OASIS                 # Data from OASIS-1
│   │   │   ├── LR                # 1mm images
│   │   │   └── MSK               # segmentations of LR data
│   │   ├── HCP                   # data from HCP database
│   │   │   ├── LR                # 1mm images
│   │   │   ├── HR                # 0.7mm images
│   │   │   └── MSK               # segmentations of HR data
│   ├── brain_simulated_t1w_mri   # simulated data
│   │   ├── HR_img                # 0.7mm images
│   │   ├── HR_msk                # segmentations of HR data
│   │   └── LR_img                # 1mm images
│   │   └── ... 
│   └── ... 
├── dataset_tio.py
├── trainer_org.py
├── trainer_gan.py
├── predict.py
├── calculate_scores.py
└── ...
...
```