# CT_NVAE

Physics-informed nouveau variational autoencoder for reconstruction in sparse computed tomography.

## Background

This code integrates the [physics-informed variational autoencoder](https://arxiv.org/abs/2211.00002) with the [nouveau variational autoencoder](https://arxiv.org/abs/2007.03898).

## Installation

Navigate to the directory where you want the code to reside and clone the repository.
```
git clone https://github.com/gigantocypris/CT_NVAE.git
```
If the repository is private, use the SSH link instead
```
git clone git@github.com:gigantocypris/CT_NVAE.git
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
```

Installing MPI in the `tomopy` environment:
```
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```

Deactivate the `tomopy` environment:
```
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

You can alternatively use the following command to install on NERSC:
```
module load pytorch/2.0.1
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

Installation is complete! Deactivate and activate the conda environment to use the newly installed dependencies

To exit the conda environment:
```
conda deactivate
```

# Covid CT Dataset Preparation
Make sure your dataset is organized as following
- data
    - pre_processed
        - Covid_CT_1.nii
        - Covid_CT_2.nii
        - Covid_CT_3.nii
    - input_npy
    - sinogram_npy
    - figures


# Synthetic Dataset Preparation

## Small Dataset Preparation on an Interactive Node

Activate the `tomopy` environment:
```
module load python
conda activate tomopy
```

Create a working directory `{WORKING_DIR}` (e.g. `$SCRATCH/output_CT_NVAE` on NERSC):
```
mkdir {WORKING_DIR}
```

Create an environment variable `WORKING_DIR`
```
export WORKING_DIR={WORKING_DIR}
```

Create an environment variable `CT_NVAE_PATH` where `{CT_NVAE_PATH}` is the path to the CT_NVAE directory (`{CT_NVAE_PATH}` could be found by cd to CT_NVAE directory and typing the `pwd` command):
```
export CT_NVAE_PATH={CT_NVAE_PATH}
```

Navigate to the working directory
```
cd $WORKING_DIR
```

On NERSC, set an environment variable with your allocation `{NERSC_GPU_ALLOCATION}` (e.g. `m3562_g`)) and start an interactive session:
```
export NERSC_GPU_ALLOCATION={NERSC_GPU_ALLOCATION}
salloc -N 1 --time=60 -C gpu -A {NERSC_GPU_ALLOCATION} --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
```

Run the following to create a synthetic foam dataset of `{T}` training examples and `{V}` validation examples, saved to the current working directory (`{T}` and `{V}` must be integers):

```
srun -n 1 python $CT_NVAE_PATH/computed_tomography/create_foam_images.py -t {T} -v {V}
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
srun -n 1 python $CT_NVAE_PATH/computed_tomography/images_to_dataset.py -n {N} -d {DATASET_TYPE}
```

The `{DATASET_TYPE}` is either `train` or `valid`, and `{N}` is the number of images to process, starting from the first image of the dataset. Complete the above for both `train` and `valid`.


## Large Dataset Preparation

Activate the `tomopy` environment:
```
module load python
conda activate tomopy
```

Create a working directory `{WORKING_DIR}` (e.g. `$SCRATCH/output_CT_NVAE` on NERSC):
```
mkdir {WORKING_DIR}
```
Create an environment variable `WORKING_DIR`
```
export WORKING_DIR={WORKING_DIR}
```
Create an environment variable `CT_NVAE_PATH` where `{CT_NVAE_PATH}` is the path to the CT_NVAE directory:
`{CT_NVAE_PATH}` could be found by cd to CT_NVAE directory and type `pwd` command
```
export CT_NVAE_PATH={CT_NVAE_PATH}
```

Navigate to the working directory
```
cd $WORKING_DIR
```

Set environment variables for the number of training and validation examples to create (note that this is the number of examples per rank):
```
export NUM_TRAIN=10
export NUM_VAL=10
```

Set environment variables for the NERSC CPU and GPU allocations, e.g. `m2859` and `m2859_g` respectively:
```
export NERSC_CPU_ALLOCATION={NERSC_CPU_ALLOCATION}
export NERSC_GPU_ALLOCATION={NERSC_GPU_ALLOCATION}
```

Run the create_foam_images_slurm.sh script to create the foam images as follows, adjusting the time limit as needed:
```
sbatch -A $NERSC_CPU_ALLOCATION -t 00:02:00 $CT_NVAE_PATH/scripts/create_foam_images_slurm.sh $NUM_TRAIN $NUM_VAL $CT_NVAE_PATH
```
This script will create foam images, separately on 4 ranks.

To visualize a training example:
```
python
import numpy as np
import matplotlib.pyplot as plt
foam_imgs = np.load('foam_train_0.npy')
plt.figure()
plt.imshow(foam_imgs[0,:,:]); plt.show()
plt.savefig('foam_training_example.png')
plt.show()
quit()
```

Run the images_to_dataset_slurm.sh script to create the sinograms, sparse sinograms, and reconstructions as follows, adjusting the time limit as needed:
```
sbatch -A $NERSC_GPU_ALLOCATION -t 00:02:00 $CT_NVAE_PATH/scripts/images_to_dataset_slurm.sh $NUM_TRAIN train $CT_NVAE_PATH
sbatch -A $NERSC_GPU_ALLOCATION -t 00:02:00 $CT_NVAE_PATH/scripts/images_to_dataset_slurm.sh $NUM_VAL valid $CT_NVAE_PATH
```

Stitch the distributed datasets together:
```
python $CT_NVAE_PATH/stitch_dist_datasets.py --num_ranks 4 --dataset_type train
python $CT_NVAE_PATH/stitch_dist_datasets.py --num_ranks 4 --dataset_type valid
```

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
salloc -N 1 --time=60 -C gpu -A $NERSC_GPU_ALLOCATION --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
```

Export the following variables:
```
export EXPR_ID=test_0000
export DATASET_DIR=$SCRATCH/output_CT_NVAE
export CHECKPOINT_DIR=checkpts
```

If on NERSC, `MASTER_ADDR` should be set as follows:
```
export MASTER_ADDR=$(hostname)
```

Otherwise:
```
export MASTER_ADDR=localhost
```

Create an environmental variable `{CT_NVAE_PATH}` if you haven't yet or you started a new session:
```
export CT_NVAE_PATH={CT_NVAE_PATH}
```

Train with the foam dataset, on a single GPU to test that the code is working:
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam --batch_size 8 --epochs 10 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax --pnm 1e1
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

[P-VAE papers](https://arxiv.org/abs/2211.00002)

[NVAE paper](https://arxiv.org/abs/2007.03898)

