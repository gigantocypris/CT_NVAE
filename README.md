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
python -m pip install nibabel
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

Installation is complete! 

To exit the conda environment:
```
conda deactivate
```

# Dataset Preparation

## Using an Interactive Node on NERSC

### Setup

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

Create an environment variable `CT_NVAE_PATH` where `{CT_NVAE_PATH}` is the path to the CT_NVAE directory and add to `PYTHONPATH`:
```
export CT_NVAE_PATH={CT_NVAE_PATH}
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
```

Navigate to the working directory
```
cd $WORKING_DIR
```

On NERSC, set an environment variable with your allocation `{NERSC_GPU_ALLOCATION}` (e.g. `m3562_g`)) and start an interactive session:
```
export NERSC_GPU_ALLOCATION={NERSC_GPU_ALLOCATION}
salloc -N 1 --time=60 -C gpu -A $NERSC_GPU_ALLOCATION --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
```

### Create 3D images

Run the following to create a synthetic foam dataset of `<n>` examples; COVID lung scans can be created by replacing `foam` with `covid` in the following directions:
```
export SLURM_NTASKS=4
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_images.py -n <n> --dest images_foam --type foam
```
Images are created in `images_foam` folder in the working directory `$WORKING_DIR`.

### Create sinograms

Sinograms are created in the existing `images_foam` folder in the working directory `$WORKING_DIR` with the following commands:
```
export SLURM_NTASKS=4
cd $WORKING_DIR
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_sinograms.py --dir images_foam
```

### Training/Validation/Test Splits

Split the dataset into training, validation, and test sets, truncating the dataset to `<n>` examples:
```
cd $WORKING_DIR
python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_foam --dest dataset_foam --train 0.7 --valid 0.2 --test 0.1 -n <n>
```
The split datasets are created in the `dataset_foam` folder in the working directory `$WORKING_DIR`.

### Create the dataset

Create the dataset with the following commands, where `<num_sparse_angles>` is the number of angles to use for the sparse sinograms and `<random>` is a boolean indicating whether to use random angles or not:
```
cd $WORKING_DIR
python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_foam --sparse <num_sparse_angles> --random <random> --ring 0
```
The dataset is created in the `dataset_foam` folder in the working directory `$WORKING_DIR`.

## Large Dataset Preparation

```
module load python
conda activate tomopy
export CT_NVAE_PATH={CT_NVAE_PATH}
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p {WORKING_DIR}
cd $WORKING_DIR
export N={number of examples}
```
Prepare foam dataset:
```
sbatch $CT_NVAE_PATH/slurm/slurm_create_dataset_foam.sh $CT_NVAE_PATH $N
```

Prepare covid dataset:
```
sbatch $CT_NVAE_PATH/slurm/slurm_create_dataset_covid.sh $CT_NVAE_PATH $N
```

## Training and validating the CT_NVAE

If using NERSC, load Python if not already loaded:
```
module load python
```

Deactivate the tomopy environment if needed, and activate the CT_NVAE environment:
```
conda activate CT_NVAE
```

Navigate to the working directory and create the `checkpts` directory`:
```
cd $WORKING_DIR
mkdir -p checkpts
```

If on NERSC, start an interactive session or see [below](#running-batch-jobs-on-NERSC) for how to run longer batch jobs:
```
salloc -N 1 --time=60 -C gpu -A $NERSC_GPU_ALLOCATION --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
```

Export the following variables, where `<experiment_description>` is a unique string identifier for the experiment:
```
export EXPR_ID=<experiment_description>
export DATASET_DIR=$WORKING_DIR
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

Create an environmental variable `{CT_NVAE_PATH}` pointing to the CT_NVAE code and add to `PYTHONPATH`:
```
export CT_NVAE_PATH={CT_NVAE_PATH}
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
```

Train with the foam dataset, on a single GPU to test that the code is working:
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam --batch_size 64 --epochs 10 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax --pnm 1e1
```

Test a longer example on 4 GPUs:
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam --batch_size 64 --epochs 100 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --pnm 1e1
```

The output is saved in `$WORKING_DIR/checkpts/eval_$EXPR_ID`.

Launch Tensorboard to view results: (TODO: Currently not working on NERSC)
```
tensorboard --logdir $CHECKPOINT_DIR/eval-$EXPR_ID/
```

## Running Batch Jobs on NERSC

To run batch jobs on NERSC:
```
conda deactivate 
conda activate CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
```
Change the permissions of the loop jobs shell script to make it executable on the terminal:
```
chmod +x $CT_NVAE_PATH/slurm/loop_jobs_batches.sh
```
Running Batch Jobs with foam dataset; each job runs on an exclusive GPU node with 4 GPUs:
```
$CT_NVAE_PATH/slurm/loop_jobs_batches.sh $CT_NVAE_PATH foam
```

The default batch_sizes='8'. However, if you want to submit 4 jobs with the batch_size of 4, 8, 16, and 32 respectively:
```
batch_sizes="4 8 16 32" $CT_NVAE_PATH/slurm/loop_jobs_batches.sh $CT_NVAE_PATH foam
```
The environmental variable 'batch_sizes' can be changed based on your preference for the duration of 'loop_jobs_batches.sh'

Check job queue status and Job ID with command:
```
squeue -u $USER
```
All jobs can be canceled with command:
```
scancel -u $USER
```
Specific jobs can be canceled with command:
```
scancel ${JobID1} ${JobID2}
```
## Covid CT Dataset Preparation

We used the [TCIA COVID-19 Dataset](https://wiki.cancerimagingarchive.net/display/Public/CT+Images+in+COVID-19). The dataset consists of 650 individual CT images, with each CT image comprising 70 image slices of size 512x512. On NERSC, the raw unzipped files from this dataset are available in `/global/cfs/cdirs/m3562/users/hkim/real_data/raw`.

If not using NERSC, download the zipped `.gz` files to a folder and update the path in `preprocessing/create_images.py`.


## Brain CT Dataset Preparation

We used the[RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data) provided by Radiological Society of North America. The dataset consists of 25k CT scans (874k slices) of human brain. On NERSC, the raw files from this dataset are available in `/global/cfs/cdirs/m3562/users/hkim/brain_data/raw`.

If not using NERSC, download the dataset using kaggle CLI. You will have to join the competition to get the full dataset. Otherwise, the Kaggle command will only download the test data. 

```
kaggle competitions download -c rsna-intracranial-hemorrhage-detection
```

After downloading the RSNA Brain Dataset, you need to convert unorganized DICOM files to organized 3D .npy file. First, run following code to generate .csv file that contains important metadata of each DICOM file and save it to `$WORKING_DIR`. 
```
python $CT_NVAE_PATH/preprocessing/brain_info.py $SOURCE_DIR $WORKING_DIR
```

Then, you can use the following command to convert DICOM files to .npy files.
```
python $CT_NVAE_PATH/preprocessing/convert_brain_dataset.py $SOURCE_DIR $TARGET_DIR 
```
where `$SOURCE_DIR` is the directory where the raw DICOM files are located and `$TARGET_DIR` is the directory where the converted .npy files will be saved.

After successfully converting DICOM files to .npy files, you can create smaller dataset by using the following command.
```
python preprocessing/make_small_dataset.py $SOURCE_DIR --average_num_slice 25 --total_slice 1000 $SMALL_TARGET_DIR
```
where `$SOURCE_DIR` is the directory where the converted .npy files are located and `$SMALL_TARGET_DIR` is the directory where the smaller dataset will be saved. The code will randomly select 3D .npy files that has about the `average_num_slice` +_5 slices and save them to the `$SMALL_TARGET_DIR` until the total number of slices reaches `total_slice`. 


## Resources:

TODO: Add more resources

[P-VAE paper](https://arxiv.org/abs/2211.00002)

[NVAE paper](https://arxiv.org/abs/2007.03898)

