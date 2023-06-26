# CT_NVAE

Physics-informed nouveau variational autoencoder for reconstruction in sparse computed tomography.

## Background

This code integrates the physics-informed variational autoencoder with the nouveau variational autoencoder (arXiv:2007.03898v3).

## Installation

Navigate to the directory where you want the code to reside and clone the repository.
```
git clone https://github.com/gigantocypris/CT_NVAE.git
```

If using NERSC, use the following command to load Python:
```
module load python
```

Create conda environment with TomoPy called `CT_NVAE`: 
```
conda create --name CT_NVAE --channel conda-forge tomopy=1.14.1 python=3.9
conda activate CT_NVAE
```

If using NERSC, start an interactive session where `{NERSC_GPU_ALLOCATION}` is your GPU allocation (e.g. m3562_g):
```
salloc -N 1 --time=60 -C gpu -A {NERSC_GPU_ALLOCATION} --qos=interactive --ntasks-per-gpu=4 --cpus-per-task=32
```

Install PyTorch in the `CT_NVAE` environment:
```
conda install pytorch==2.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install the other conda dependencies:
```
conda install h5py
```

Upgrade pip:
```
python -m pip install --upgrade pip
```

Install the pip dependencies:
```
python -m pip install -r requirements.txt
```

Installation is complete!

To exit the conda environment:
```
conda deactivate
```

# Synthetic Dataset Preparation

Activate the `CT_NVAE` environment:
```
conda activate CT_NVAE
```

Create a working directory `{WORKING_DIR}` (e.g. `$SCRATCH/output_CT_NVAE` on NERSC):
```
mkdir {WORKING_DIR}
```

Create an environment variable `CT_NVAE_PATH` where `{CT_NVAE_PATH}` is the path to the CT_NVAE directory:
```
export CT_NVAE_PATH={CT_NVAE_PATH}
```

Navigate to the working directory
```
cd {WORKING_DIR}
```

Run the following to create a synthetic foam dataset of `{T}` training examples and `{V}` validation examples, saved in the subfolder `dataset_foam` of the current directory:
```
python $CT_NVAE_PATH/computed_tomography/create_foam_images.py -t {T} -v {V}
```

To visualize a training example:
```
python
import numpy as np
import matplotlib.pyplot as plt
foam_imgs = np.load('foam_training.npy')
plt.figure()
plt.imshow(foam_imgs[0,:,:]); plt.show()
plt.show()
```

To generate sinograms (project images) from the foam images, create sparse sinograms, and reconstruct from the sparse sinograms:
```
export KMP_DUPLICATE_LIB_OK=TRUE
python $NVAE_PATH/computed_tomography/images_to_dataset.py -n {N} -d {DATASET_TYPE}
```
The `{DATASET_TYPE}` is either `train` or `valid`, and `{N}` is the number of images to process, starting from the first image of the dataset.


**TODO: Convert to unit test with PyTest**
**TODO: Compare (timing and accuracy) with TorchRadon implemented by Hojune and Gary**
**TODO: Compare (timing and accuracy) with torch rotation and vmap**
Test Radon transform in PyTorch and compare with the Radon transform in TomoPy:
```
python $CT_NVAE_PATH/computed_tomography/test_forward_physics.py 
```

## Training and validating the CT_NVAE

If using NERSC, load Python:
```
module load python
```

Activate the CT_NVAE environment:
```
conda activate CT_NVAE
```

Navigate to the `CT_NVAE` directory and create the following directories:
```
cd $CT_NVAE_PATH
mkdir data
mkdir checkpts
```

Navigate to the working directory:
```
cd {WORKING_DIR}
```

If on NERSC, start an interactive session or see [below](#running-batch-jobs-on-NERSC) for how to run longer batch jobs:
```
salloc -N 1 --time=60 -C gpu -A {NERSC_GPU_ALLOCATION} --qos=interactive --ntasks-per-gpu=4 --cpus-per-task=32
```

Export the following variables:
```
export EXPR_ID=test_0000
export DATA_DIR=data
export CHECKPOINT_DIR=checkpts
export MASTER_ADDR=localhost
```

If on NERSC, `MASTER_ADDR` should be set as follows:
```
export MASTER_ADDR=$(hostname)
```

Train with the foam dataset, on a single GPU to test that the code is working:
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam --batch_size 8 --epochs 10 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax
```

Test a longer example on 4 GPUs:
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam --batch_size 8 --epochs 100 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --pnm 1e1
```

Launch Tensorboard to view results:
```
tensorboard --logdir $CHECKPOINT_DIR/eval-$EXPR_ID/
```

**TODO: How to view Tensorboard while training**

## Running Batch Jobs on NERSC

To run a batch job on NERSC:
```
sbatch scripts\train_single_node.sh
```

**TODO: Figure out multinode training on NERSC**

## TODO

version numbers for all packages in the install directions

Warning:
/global/homes/X/XX/.conda/envs/NVAE_2/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py:2388: UserWarning: torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.

Licensing: figure out what license to use for modified NVIDIA files

Add real dataset

Try different datasets

Try different model distributions (currently using the RelaxedBernoulli distribution)

Train and validate with different datasets

Allow for training with the full object reconstructions and compare sample complexity

Add a ring artifact and see if the neural network can remove

Try real dataset, use no-reference metrics to evaluate

## Resources:

P-VAE papers

NVAE paper