# CT_NVAE

Physics-informed nouveau variational autoencoder for reconstruction in sparse computed tomography.

## Background

This code integrates the [physics-informed variational autoencoder](https://arxiv.org/abs/2211.00002) with the [nouveau variational autoencoder](https://arxiv.org/abs/2007.03898).

## Installation

Navigate to the directory where you want the code to reside and clone the repository.
```
git clone https://github.com/gigantocypris/CT_NVAE.git
```

If using NERSC, use the following command to load Python:
```
module load python
```

Create conda environment with TomoPy called `tomopy` and activate: 
```
conda create --name tomopy --channel conda-forge tomopy=1.14.1 python=3.9
conda activate tomopy
```

Install PyTorch with only CPU support in the `tomopy` environment:
```
python -m pip install xdesign
python -m pip install kornia
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
conda deactivate
```

The `tomopy` environment will be used for pre-processing the data. We will create another environment for training the CT_NVAE because the requirements for PyTorch GPU conflict with the requirements for TomoPy.


Create conda environment called `CT_NVAE`: 
```
conda create --name CT_NVAE python=3.9
conda activate CT_NVAE
```

Install PyTorch in the `CT_NVAE` environment:
```
conda install pytorch==2.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Test PyTorch install:
```
python
import torch 
print(torch.cuda.is_available())
print(torch.cuda.device_count())
quit()
```

Output should be `True` and `1` if on a NERSC login node.

Install the other conda dependencies:
```
conda install h5py
```

Upgrade pip:
```
python -m pip install --upgrade pip
```

Navigate to the CT_NVAE directory.

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

Activate the `tomopy` environment:
```
module load python
conda activate tomopy
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

For large dataset creation on NERSC, start an interactive session, where `{NERSC_GPU_ALLOCATION}` is your GPU allocation (e.g. m3562_g):
```
salloc -N 1 --time=60 -C gpu -A {NERSC_GPU_ALLOCATION} --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
```

Run the following to create a synthetic foam dataset of `{T}` training examples and `{V}` validation examples, saved to the current working directory:
```
python $CT_NVAE_PATH/computed_tomography/create_foam_images.py -t {T} -v {V}
```

To visualize a training example:
```
python
import numpy as np
import matplotlib.pyplot as plt
foam_imgs = np.load('foam_train.npy')
plt.figure()
plt.imshow(foam_imgs[0,:,:]); plt.show()
plt.savefig('foam_training_example.png')
plt.show()
quit()
```

To generate sinograms (project images) from the foam images, create sparse sinograms, and reconstruct from the sparse sinograms, saved in the subfolder `dataset_foam` of the current directory:
```
python $CT_NVAE_PATH/computed_tomography/images_to_dataset.py -n {N} -d {DATASET_TYPE}
```
The `{DATASET_TYPE}` is either `train` or `valid`, and `{N}` is the number of images to process, starting from the first image of the dataset. Complete the above for both `train` and `valid`.

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

Navigate to the working directory and create the `checkpts` directory`:
```
cd {WORKING_DIR}
mkdir checkpts
```

If on NERSC, start an interactive session or see [below](#running-batch-jobs-on-NERSC) for how to run longer batch jobs:
```
salloc -N 1 --time=60 -C gpu -A {NERSC_GPU_ALLOCATION} --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
```

Export the following variables:
```
export EXPR_ID=test_0000
export DATASET_DIR=$SCRATCH/output_CT_NVAE
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

## Running Batch Jobs on NERSC

To run a batch job on NERSC:
```
sbatch $CT_NVAE_PATH/scripts/train_single_node.sh
```
## Resources:

P-VAE papers

NVAE paper