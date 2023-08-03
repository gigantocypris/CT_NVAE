# CT-NVAE

Physics-informed nouveau variational autoencoder (CT-NVAE) for reconstruction in sparse computed tomography.

# Table of Contents

## Background

This code integrates the [physics-informed variational autoencoder](https://arxiv.org/abs/2211.00002) with the [nouveau variational autoencoder](https://arxiv.org/abs/2007.03898). The particular application considered in this repository is sparse x-ray computed tomography (CT) though other computational imaging applications are possible through modification of the forward physics. The code uses PyTorch.

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
python -m pip install pandas
python -m pip install pydicom
python -m pip install tqdm
```

The following packages need to be installed if you are downloading and preprocessing the CT brain data on your own computing setup:
```
python -m pip instal bleach
python -m pip install kaggle
```

Install MPI in the `tomopy` environment:
```
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```

Deactivate the `tomopy` environment:
```
conda deactivate
```

The `tomopy` environment will be used for pre-processing the data. We will create another environment for training the CT_NVAE because the requirements for PyTorch with GPU conflict with the requirements for TomoPy.


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

# Downloading Data

## CT Lung Scans of COVID Patients

We used the [TCIA COVID-19 Dataset](https://wiki.cancerimagingarchive.net/display/Public/CT+Images+in+COVID-19). The dataset consists of 650 individual CT images, with each CT image comprising 70 image slices of size 512x512. On NERSC, the raw unzipped files from this dataset are available in `/global/cfs/cdirs/m3562/users/hkim/real_data/raw`.

If using NERSC, set an environment variable for the path to the raw data:
```
export COVID_RAW_DATA=/global/cfs/cdirs/m3562/users/hkim/real_data/raw
```

If not using NERSC, download the zipped `.gz` files to a folder set an environment variable for the path to the raw data:
```
export COVID_RAW_DATA=<path_to_raw_data>
```

## CT Brain Scans

We used the[RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data) provided by Radiological Society of North America. The dataset consists of 25k CT scans (874k slices) of human brain. On NERSC, the raw files from this dataset are available in `/global/cfs/cdirs/m3562/users/hkim/brain_data/raw`.

If not using NERSC, download the dataset using Kaggle CLI. You will have to join the competition to get the full dataset. Otherwise, the Kaggle command will only download the test data. 

```
kaggle competitions download -c rsna-intracranial-hemorrhage-detection
```

After downloading the RSNA Brain Dataset, you need to convert unorganized DICOM files to organized 3D `.npy` files (each .npy file has the 2D slices from a single scan). We provide `.csv` files that contains important metadata of each DICOM file. You can use these csv files to convert DICOM files to organized `.npy` files. The link to the csv files is [here](https://drive.google.com/drive/folders/1kcci7GK4M5-etTD8zf75z7GJhLYTFEFn?usp=sharing). On NERSC, the `.csv` files are available at `/global/cfs/cdirs/m3562/users/hkim/brain_data/brain_merged_info.csv` and `/global/cfs/cdirs/m3562/users/hkim/brain_data/instance_thickness.csv`. Alternatively, you can generate the `.csv` files by yourself using the following commands:
```
conda activate tomopy
python $CT_NVAE_PATH/preprocessing/brain_create_CSV.py $SOURCE_DIR $TEMP_CSV_DIR  
```
 `$SOURCE_DIR` is the directory where the raw DICOM files are located and `$TEMP_CSV_DIR` is the directory where the intermediary csv files will be saved. After creating all the intermediary csv files, you can merge them and sort them by using the following command: 
```
python $CT_NVAE_PATH/preprocessing/brain_merge_and_sort_CSV.py $TEMP_CSV_DIR $FINAL_CSV_PATH $THICKNESS
```
`$FINAL_CSV_PATH` is the path where the final `brain_merged_info.csv` file will be saved. `$THICKNESS` is the path where the `instance_thickness.csv` file will be saved.

After downloading or generating the csv file, you can convert DICOM files to organized 3D `.npy` files; detailed in the steps below.

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

#### Synthetic Foam Images or CT Lung Scans of COVID Patients

Run the following to create a synthetic foam or lung scan dataset of `<n>` examples.
For synthetic foam images, set the following environment variable:
```
export DATA_TYPE=foam
```

For CT lung scans of COVID patients, set the following environment variable:
```
export DATA_TYPE=covid
```

Then run the following command:
```
export SLURM_NTASKS=4
cd $WORKING_DIR
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_images.py -n <n> --dest images_$DATA_TYPE --type $DATA_TYPE
```
Images are created in `images_$DATA_TYPE` folder in the working directory `$WORKING_DIR`.

#### CT Brain Scans

To process the CT brain scans, you need to convert the DICOM files to organized 3D `.npy` files. You can do this by setting the following environment variables and running the following command: 
```
export CSV_PATH=<path to brain_merged_info.csv>
export DCM_PATH=<folder containing the raw DICOM files>
export OUTPUT_PATH=<path to the output folder to be created where the 3D .npy files will reside>
export THICKNESS=<path to instance_thickness.csv>
export NUM_INSTANCE=<number of 3D .npy files to create>

cd $WORKING_DIR
python $CT_NVAE_PATH/preprocessing/preprocess_brain.py $CSV_PATH $DCM_PATH $OUTPUT_PATH $THICKNESS $NUM_INSTANCE
```
If you want to convert all the instances, you can set `$NUM_INSTANCE` to 21744.

For example, on NERSC, you can do the following to use a random 50 3D images selection of training examples from the Kaggle dataset:
```
export CSV_PATH=/global/cfs/cdirs/m3562/users/hkim/brain_data/brain_merged_info.csv
export DCM_PATH=/global/cfs/cdirs/m3562/users/hkim/brain_data/raw/stage_2_train/
export OUTPUT_PATH=$SCRATCH/output_CT_NVAE/output_brain/50_instance_npy/
export THICKNESS=/global/cfs/cdirs/m3562/users/hkim/brain_data/instance_thickness.csv
export NUM_INSTANCE=50
python $CT_NVAE_PATH/preprocessing/preprocess_brain.py $CSV_PATH $DCM_PATH $OUTPUT_PATH $THICKNESS $NUM_INSTANCE
```

### Create sinograms

Create an environment variable pointing to the folder containing the images:
For the foam images or COVID lung scans, you can do:
```
export IMAGES_DIR=images_$DATA_TYPE
```

For the CT brain scans, you can do:
```
export IMAGES_DIR=$OUTPUT_PATH
```

Sinograms are created in the existing `$IMAGES_DIR` folder in the working directory `$WORKING_DIR` with the following commands:
```
export SLURM_NTASKS=4
cd $WORKING_DIR
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_sinograms.py --dir $IMAGES_DIR
```

### Training/Validation/Test Splits

Split the dataset into training, validation, and test sets, truncating the dataset to `<n>` examples:
```
cd $WORKING_DIR
python $CT_NVAE_PATH/preprocessing/create_splits.py --src $IMAGES_DIR --dest dataset_foam --train 0.7 --valid 0.2 --test 0.1 -n <n>
```
The split datasets are created in the `dataset_foam` folder in the working directory `$WORKING_DIR`.

### Create the dataset
Export an environment variable `DATASET_ID` to identify the dataset, for example:
```
export DATASET_ID=$DATA_TYPE
```

Create the dataset with the following commands, where `<num_sparse_angles>` is the number of angles to use for the sparse sinograms and `<random>` is a boolean indicating whether to use random angles or not:
```
cd $WORKING_DIR
python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_$DATASET_ID --sparse <num_sparse_angles> --random <random> --ring 0 --pnm 1e3
```
The dataset is created in the `dataset_$DATASET_ID` folder in the working directory `$WORKING_DIR`.

The option `--pnm` is for the Poisson noise multiplier. Essentially, a higher value means higher signal-to-noise ratio (SNR). The option `--ring` is for adding a ring artifact. A higher number increases the strength of the ring artifact. There is a different artifact added to each 3D image. The CT-NVAE will later attempt to remove this artifact. if `--model_ring_artifact` is passed to the training script.  

## Using the Slurm scheduler on NERSC

XXX Need to test from here, VG 8/1/2023
TODO: pass the amount of time as argument to sbatch

To create a larger dataset, it is recommended to use sbatch jobs with the Slurm scheduler.
```
module load python
conda activate tomopy
export CT_NVAE_PATH=<path to CT_NVAE repository>
export WORKING_DIR=<path to working directory>
mkdir -p $WORKING_DIR
cd $WORKING_DIR
export NUM_EXAMPLES=<number of 3D examples to create for the dataset>
```

Prepare the synthetic foam dataset with the following commands:
```
cd $WORKING_DIR
export DATA_TYPE=foam
export IMAGE_ID=$DATA_TYPE
export DATASET_ID=$DATA_TYPE
export NUM_SPARSE_ANGLES=<number of projection angles to use>
export RANDOM=<boolean indicating whether to use random angles or not>

sbatch $CT_NVAE_PATH/slurm/create_dataset_foam.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM
```

Prepare a dataset of CT lung scans of COVID patients following the directions above but replacing the `$DATA_TYPE`:
```
export DATA_TYPE=foam
```

Prepare a dataset of CT brain scans:
```
TODO
```

## Training and validating the CT_NVAE

### Using an interactive node on NERSC

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

To have the CT-NVAE attempt to remove the ring artifact, additionally pass `--model_ring_artifact` to `train.py`.

### Using the Slurm scheduler on NERSC

XXX TODO write script with the preempt queue

To run batch jobs on NERSC:
```
conda deactivate 
conda activate CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
```
Change the permissions of the loop jobs shell script to make it executable on the terminal:
```
chmod +x $CT_NVAE_PATH/slurm/loop_jobs.sh
```
Running Batch Jobs with foam dataset; each job runs on an exclusive GPU node with 4 GPUs:
```
./$CT_NVAE_PATH/slurm/loop_jobs.sh $CT_NVAE_PATH foam
```

The default batch_sizes='8'. However, if you want to submit 4 jobs with the batch_size of 4, 8, 16, and 32 respectively:
```
batch_sizes="4 8 16 32" ./$CT_NVAE_PATH/slurm/loop_jobs.sh $CT_NVAE_PATH foam
```
The environmental variable 'batch_sizes' can be changed based on your preference for the duration of 'loop_jobs.sh'

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

## Evaluating the CT-NVAE

## Evaluating the CT-NVAE with 2D Covid Classifier
Step0: Imitate the training process of the CT-NVAE to create the reconstructed data. `RECON_PATH` is the path that contains _recon.npy and _label.npy files.
```
conda deactivate
module load python
conda activate tomopy
export WORKING_DIR=/pscratch/sd/h/hojunek/2DCOVID
export CT_NVAE_PATH=/global/homes/h/hojunek/CT_NVAE
export NERSC_GPU_ALLOCATION=m3562_g
export SLURM_NTASKS=4
export DATA_TYPE=covid2D
export IMG_PATH=small_SARS_npy_1k
export DATASET_PATH=dataset_small_SARS_npy_1k
cd $WORKING_DIR
salloc -N 1 --time=60 -C gpu -A $NERSC_GPU_ALLOCATION --qos=interactive --ntasks-per-gpu=4 --cpus-per-task=32
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_images.py -n 1000 --dest $IMG_PATH --type $DATA_TYPE
export PREPROCESSED_PATH=/pscratch/sd/h/hojunek/2DCOVID/small_SARS_npy_1k
export RECON_PATH=/pscratch/sd/h/hojunek/2DCOVID/small_SARS_npy_1k_recon
python $CT_NVAE_PATH/classifiers/COVID/code/fake_recon.py $PREPROCESSED_PATH $RECON_PATH
```

Step1: Define the path to convert the reconstructed data to png files. `RECON_PNG_PATH` is the path that contains _recon.png files and `recon_label.txt`.
```
export RECON_PNG_PATH=/pscratch/sd/h/hojunek/2DCOVID/small_SARS_npy_1k_test_preprocess
python $CT_NVAE_PATH/classifiers/COVID/code/test_recon_preprocess.py $RECON_PATH $RECON_PNG_PATH
```

Step2: Run the 2D classifier to evaluate the reconstructed data.
```
python $CT_NVAE_PATH/classifiers/COVID/code/test_recon.py $RECON_PNG_PATH 
```


TODO: visualize with the analysis script

```
module load python
conda activate tomopy
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
cd $WORKING_DIR
python $CT_NVAE_PATH/metrics/analyze_training_results.py 
```

## Citation

If you find this repository useful to you in your research, please cite our paper:

```
TODO
```

## Further Resources:

### Previous Papers/Repositories

[P-VAE paper](https://arxiv.org/abs/2211.00002)

[NVAE paper](https://arxiv.org/abs/2007.03898)

### NERSC

[Queues and Charges at NERSC](https://docs.nersc.gov/jobs/policy/)

[NERSC preempt queue](https://docs.nersc.gov/jobs/examples/#preemptible-jobs)

TODO: Add more resources