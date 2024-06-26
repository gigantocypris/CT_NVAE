Start with:
module load python
conda activate tomopy

# Installing MPI in the conda environment `tomopy`
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py

export NUM_TRAIN=10
export NUM_VAL=10
export NERSC_CPU_ALLOCATION=m2859
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
sbatch -A $NERSC_CPU_ALLOCATION -t 00:02:00 $CT_NVAE_PATH/scripts/create_foam_images_slurm.sh $NUM_TRAIN $NUM_VAL $CT_NVAE_PATH
Submitted batch job 10799095
DONE

export NERSC_GPU_ALLOCATION=m2859_g
sbatch -A $NERSC_GPU_ALLOCATION -t 00:02:00 $CT_NVAE_PATH/scripts/images_to_dataset_slurm.sh $NUM_TRAIN train $CT_NVAE_PATH
Submitted batch job 10800187
DONE

sbatch -A $NERSC_GPU_ALLOCATION -t 00:02:00 $CT_NVAE_PATH/scripts/images_to_dataset_slurm.sh $NUM_VAL valid $CT_NVAE_PATH
Submitted batch job 10800191
DONE

Stitch the dataset:
cd $SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
python $CT_NVAE_PATH/stitch_dist_datasets.py --num_ranks 4 --dataset_type train
python $CT_NVAE_PATH/stitch_dist_datasets.py --num_ranks 4 --dataset_type valid

# Running the loop
. $CT_NVAE_PATH/scripts/loop_jobs.sh

### Notes on Covid dataset

scp (with sshproxy) command to upload a folder to NERSC:
```
scp -r -O /Users/vganapa1/Downloads/CT-Covid-19 vidyagan@saul-p1.nersc.gov:/pscratch/sd/v/vidyagan
```

Commands run:
```
export NERSC_GPU_ALLOCATION=m3562_g
export NERSC_CPU_ALLOCATION=m3562

export CT_NVAE_PATH=$SCRATCH/CT_NVAE
cd $SCRATCH/output_CT_NVAE
python $SCRATCH/CT_NVAE/computed_tomography/create_real_dataset.py --dir dataset_covid -n 2 -d train
python $SCRATCH/CT_NVAE/computed_tomography/create_real_dataset.py --dir dataset_covid -n 1 -d valid
```


## July 14, 2023

Refactoring entire pipeline, inclusion of ring artifact

`tomopy` environment:
1. Get data: Either (1) convert real data to npy or (2) create 3D foam data
computed_tomography/create_images.py
Make a folder: images_foam
2. Go through all examples one by one and create a corresponding sinogram
Put in the folder images_foam
computed_tomography/create_sinograms.py
3. Split into training/test/validate
split within the folder images_foam
scripts/create_splits.py
4. Create a dataset from each of the splits
this will be in the newly created folder dataset_foam; each 3D example has a common identifier for ring artifact removal
computed_tomography/create_dataset.py


`CT_NVAE` environment:
```
module load python
conda activate CT_NVAE
salloc -N 1 --time=120 -C gpu -A m3562_g --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
cd output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export EXPR_ID=test_0000_foam2
export DATASET_DIR=$SCRATCH/output_CT_NVAE
export CHECKPOINT_DIR=checkpts
export MASTER_ADDR=$(hostname)
```

Single GPU:
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam2 --batch_size 8 --epochs 10 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax --pnm 1e1
```

Multi-GPU:
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam2 --batch_size 32 --epochs 100 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --pnm 1e1
```

5. Refactor the CT_NVAE code to allow any CT dataset in (remove all the old datasets), removal of ring artifact, option for the output distribution to be Gaussian (need an extra dimension in the output) or Bernoulli
6. Go through and make the entire pipeline sbatch-able

## July 19, 2023

### Full pass through the pipeline with the foam images:

Start with the `tomopy` environment:
```
module load python
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
conda activate tomopy
salloc -N 1 --time=120 -C gpu -A m3562_g --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
```

1. Create the foam images
```
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export SLURM_NTASKS=4
cd $WORKING_DIR
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_images.py -n 64 --dest images_foam --type foam
```
Images are created in `images_foam` folder in the working directory `$WORKING_DIR`.

2. Create the sinograms
```
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export SLURM_NTASKS=4
cd $WORKING_DIR
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_sinograms.py --dir images_foam
```
Sinograms are created in the existing `images_foam` folder in the working directory `$WORKING_DIR`.

3. Split into training/test/validate
```
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_foam --dest dataset_foam2 --train 0.7 --valid 0.2 --test 0.1 -n 64
```
The split datasets are created in the `dataset_foam2` folder in the working directory `$WORKING_DIR`.

Create ring artifact dataset:
```
python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_foam --dest dataset_foam_ring --train 0.7 --valid 0.2 --test 0.1 -n 64
```

4. Create the dataset
```
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
cd $WORKING_DIR
python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_foam2 --sparse 20 --random True --ring 0
```
The dataset is created in the `dataset_foam2` folder in the working directory `$WORKING_DIR`.

Create ring artifact dataset:
```
python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_foam_ring --sparse 20 --random True --ring 0.3
```

5. Train the model.
First exit the interactive session, `conda deactivate`, and start a new one with the `CT_NVAE` environment:
```
module load python
conda activate CT_NVAE
salloc -N 1 --time=120 -C gpu -A m3562_g --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
```

Set environment variables for training:
```
export EXPR_ID=test_0000_foam2_wandb
export DATASET_DIR=$SCRATCH/output_CT_NVAE
export CHECKPOINT_DIR=checkpts
export MASTER_ADDR=$(hostname)
```

Single GPU training to test everything is working:
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam2 --batch_size 64 --epochs 10 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax --pnm 1e1
```

Multi-GPU training:
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam2 --batch_size 64 --epochs 100 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --pnm 1e1
```

### Full pass with the COVID data

Start with the `tomopy` environment:
```
module load python
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
conda activate tomopy
salloc -N 1 --time=120 -C gpu -A m3562_g --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
```

1. Create the COVID images
```
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export SLURM_NTASKS=4
cd $WORKING_DIR
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_images.py -n 64 --dest images_covid --type covid
```
Images are created in `images_covid` folder in the working directory `$WORKING_DIR`.

2. Create the sinograms
```
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export SLURM_NTASKS=4
cd $WORKING_DIR
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_sinograms.py --dir images_covid
```
Sinograms are created in the existing `images_covid` folder in the working directory `$WORKING_DIR`.

3. Split into training/test/validate
```
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_covid --dest dataset_covid2 --train 0.7 --valid 0.2 --test 0.1 -n 64
```
The split datasets are created in the `dataset_covid2` folder in the working directory `$WORKING_DIR`.

4. Create the dataset
```
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
cd $WORKING_DIR
python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_covid2 --sparse 20 --random True --ring 0
```
The dataset is created in the `dataset_covid2` folder in the working directory `$WORKING_DIR`.


5. Train the model.
First exit the interactive session, `conda deactivate`, and start a new one with the `CT_NVAE` environment:
```
module load python
conda activate CT_NVAE
salloc -N 1 --time=120 -C gpu -A m3562_g --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
```

Set environment variables for training:
```
export EXPR_ID=test_0000_covid3
export DATASET_DIR=$SCRATCH/output_CT_NVAE
export CHECKPOINT_DIR=checkpts
export MASTER_ADDR=$(hostname)
```

Single GPU training to test everything is working:
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset covid2 --batch_size 8 --epochs 10 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax --pnm 1e1
```

Multi-GPU training:
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset covid2 --batch_size 8 --epochs 10 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --pnm 1e1
```


### Full pass with the brain data
TODO

Raw data files are available here:
/global/cfs/cdirs/m3562/users/hkim/brain_data/raw

```
 export SOURCE_DIR={SOURCE_DIR}
 export TARGET_DIR={TARGET_DIR}
```
You can use the `computed_tomography/preprocess_brain_data.py` script provided to accomplish this. Set `small` into `True` to make a small dataset.(100 Patients) The number of files that will be processed is given by the -n flag. The -v flag is optional and will print out .png visualizations of all the images and sinograms in the dataset. Only use the -v flag for a small dataset.

```
python $CT_NVAE_PATH/computed_tomography/preprocess_brain_data.py $SOURCE_DIR $TARGET_DIR -small True -n 100
```

## Wandb

``` 
conda activate CT_NVAE
python -m pip install wandb
wandb login
https://wandb.ai/gigantocypris
```

## July 24, 2023

### Full pass through the pipeline with the foam images and ring artifact

Start with the `tomopy` environment:
```
module load python
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
conda activate tomopy
salloc -N 1 --time=120 -C gpu -A m3562_g --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
```

1. Use foam images already created in `images_foam` folder in the working directory `$WORKING_DIR`.

2. Use sinograms already created in the existing `images_foam` folder in the working directory `$WORKING_DIR`.

3. Split into training/test/validate
```
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_foam --dest dataset_foam_ring --train 0.7 --valid 0.2 --test 0.1 -n 64
```
The split datasets are created in the `dataset_foam_ring` folder in the working directory `$WORKING_DIR`.


4. Create the dataset
```
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
cd $WORKING_DIR
python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_foam_ring --sparse 20 --random True --ring 0.05
```
The dataset is created in the `dataset_foam_ring` folder in the working directory `$WORKING_DIR`.


5. Train the model.
First exit the interactive session, `conda deactivate`, and start a new one with the `CT_NVAE` environment:
```
module load python
conda activate CT_NVAE
salloc -N 1 --time=120 -C gpu -A m3562_g --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
```

Set environment variables for training:
```
export EXPR_ID=test_0000_foam2_ring
export DATASET_DIR=$SCRATCH/output_CT_NVAE
export CHECKPOINT_DIR=checkpts
export MASTER_ADDR=$(hostname)
```

Single GPU training to test everything is working (not considering ring artifact):
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam_ring --batch_size 64 --epochs 10 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax --pnm 1e1 --save_interval 20
```

Single GPU training to test everything is working (considering ring artifact):
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam_ring --batch_size 64 --epochs 10 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax --pnm 1e1 --save_interval 20 --model_ring_artifact
```

Multi-GPU training, considering ring artifact
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam_ring --batch_size 64 --epochs 10 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --pnm 1e1 --save_interval 20 --model_ring_artifact
```

Evaluation run (loads previously trained Multi-GPU and only does evaluation of valid queue at end), adds --cont_training and sets epochs to 0:
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam_ring --batch_size 64 --epochs 0 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --pnm 1e1 --save_interval 20 --model_ring_artifact --cont_training
```


Multi-GPU training, doesn't consider ring artifact
```
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam_ring --batch_size 64 --epochs 10 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --pnm 1e1 --save_interval 20
```

### July 27, 2023

analyze results from a training run:

```
module load python
conda activate tomopy
salloc -N 1 --time=120 -C gpu -A m3562_g --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
python $CT_NVAE_PATH/metrics/analyze_training_results.py 
```

### July 31, 2023

Debugging brain pipeline:
module load python
cd $SCRATCH
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
python -m pip install pandas
python -m pip install pydicom
python -m pip install tqdm
salloc -N 1 --time=120 -C gpu -A m3562_g --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32

# export NERSC_GPU_ALLOCATION=m3562_g
# export NERSC_CPU_ALLOCATION=m3562
##  Start Interactive Node
# salloc -N 1 --time=120 -C gpu -A $NERSC_GPU_ALLOCATION --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32


export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE

=====================Convert DCM-NPY========================
export CSV_PATH=/global/cfs/cdirs/m3562/users/hkim/brain_data/brain_merged_info.csv
export DCM_PATH=/global/cfs/cdirs/m3562/users/hkim/brain_data/raw/stage_2_train/
export OUTPUT_PATH=$SCRATCH/output_CT_NVAE/output_brain/10_instance_npy/
export THICKNESS=/global/cfs/cdirs/m3562/users/hkim/brain_data/instance_thickness.csv
export NUM_INSTANCE=10
echo $CSV_PATH $DCM_PATH $OUTPUT_PATH $THICKNESS $NUM_INSTANCE
cd $WORKING_DIR
python $CT_NVAE_PATH/preprocessing/preprocess_brain.py $CSV_PATH $DCM_PATH $OUTPUT_PATH $THICKNESS $NUM_INSTANCE

repeat for 50 example dataset:
export OUTPUT_PATH=$SCRATCH/output_CT_NVAE/output_brain/50_instance_npy/
export NUM_INSTANCE=50

========================PREPROCESS===========================
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH

export SLURM_NTASKS=4
echo $WORKING_DIR, $CT_NVAE_PATH, $PYTHONPATH, $NERSC_CPU_ALLOCATION, $NERSC_GPU_ALLOCATION, $SLURM_NTASKS
cd $WORKING_DIR

export OUTPUT_PATH=$SCRATCH/output_CT_NVAE/output_brain/50_instance_npy/
export DATASET_NAME=brain_50_instance
# Create Sinogram
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_sinograms.py --dir $OUTPUT_PATH
# Split into train/val/test
python $CT_NVAE_PATH/preprocessing/create_splits.py --src $OUTPUT_PATH --dest dataset_$DATASET_NAME --train 0.7 --valid 0.2 --test 0.1
# Create Dataset
python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_$DATASET_NAME --sparse 45 --random True --ring 0
==============================================================
VISUALIZE in the images folder:

```
python
import numpy as np
import matplotlib.pyplot as plt
a = np.load('/pscratch/sd/v/vidyagan/output_CT_NVAE/output_brain
/50_instance_npy/ID_0a630be69b.npy')
i=0;plt.figure();plt.imshow(a[i]);plt.colorbar();plt.savefig('brain'+str(i)+'.png')
```

========================TRAIN===========================
exit conda environment and start a new CT_NVAE environment:

module load python
conda activate CT_NVAE
salloc -N 1 --time=120 -C gpu -A m3562_g --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32

export WORKING_DIR=$SCRATCH/output_CT_NVAE
export NERSC_GPU_ALLOCATION=m3562_g
export CHECKPOINT_DIR=checkpts
export MASTER_ADDR=$(hostname)
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export DATASET_DIR=$WORKING_DIR

export DATASET_NAME=brain_50_instance
export EXPR_ID=test_$DATASET_NAME

echo $WORKING_DIR, $NERSC_GPU_ALLOCATION, $EXPR_ID, $CHECKPOINT_DIR, $MASTER_ADDR, $CT_NVAE_PATH, $PYTHONPATH

cd $WORKING_DIR

# Running train.py (single GPU)
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset $DATASET_NAME --batch_size 16 --epochs 10 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax --pnm 1e1

# Running train.py (4 GPUs)
change to `--num_process_per_node 4`

## Changing permissions on NERSC

```
chmod 775 myfile.txt # For a file
chmod -R 775 my_folder # For a Folder
```
- first 7 means all permission to owner
- second 7 means all permission to the group
- third 5 means read and execute permission to everyone else on NERSC

## Testing Slurm scripts, August 2, 2023


For Foam:
export NERSC_GPU_ALLOCATION=m3562_g

module load python
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR
export NUM_EXAMPLES=50

export DATA_TYPE=foam
export IMAGE_ID=foam_slurm
export DATASET_ID=foam_slurm
export NUM_SPARSE_ANGLES=90
export RANDOM=True

sbatch --time=00:05:00 -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM
Submitted batch job 13133277

For Covid:
export DATA_TYPE=covid
export IMAGE_ID=covid_slurm
export DATASET_ID=covid_slurm
export COVID_RAW_DATA=/global/cfs/cdirs/m3562/users/hkim/real_data/raw

sbatch --time=00:10:00 -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM

Submitted batch job 13131807

For brain:
export DATA_TYPE=brain
export IMAGE_ID=brain_slurm
export DATASET_ID=brain_slurm

sbatch --time=00:10:00 -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM

Submitted batch job 13131826

Testing "Training and validating the CT_NVAE":
export NERSC_GPU_ALLOCATION=m3562_g
module load python
conda activate CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE

cd $WORKING_DIR

salloc -N 1 --time=120 -C gpu -A $NERSC_GPU_ALLOCATION --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32

export EXPR_ID=testing_foam_interactive
export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=checkpts
export MASTER_ADDR=$(hostname)
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH

export DATASET_ID=foam_ring
export NUM_GPU=1

python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset $DATASET_ID --batch_size 64 --epochs 10 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 1 --ada_groups --num_process_per_node $NUM_GPU --use_se --res_dist --fast_adamax --pnm 1e1 --save_interval 20

## August 3, 2023

export NERSC_GPU_ALLOCATION=m3562_g
module load python
conda activate CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE

cd $WORKING_DIR

export EXPR_ID=testing_foam_interactive
export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=checkpts
export MASTER_ADDR=$(hostname)
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH

export DATASET_ID=foam_ring

tensorboard --logdir $CHECKPOINT_DIR/eval-$EXPR_ID/

salloc -N 1 --time=120 -C gpu -A $NERSC_GPU_ALLOCATION --qos=interactive --ntasks-per-gpu=1 --cpus-per-task
=32

### Starting from fresh terminal and running job as a batch script:

module load python
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts

export NERSC_GPU_ALLOCATION=m3562_g
export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
export MASTER_ADDR=$(hostname)
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH

export DATASET_ID=foam_slurm

export BATCH_SIZE=8
export EPOCHS=10
export SAVE_INTERVAL=20

sbatch -A $NERSC_GPU_ALLOCATION -t 00:10:00 $CT_NVAE_PATH/slurm/train_single_node.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL
Submitted batch job 13203549 OOM

Submitted batch job 13204372 Success

Visualization:


module load python
conda activate tomopy

export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
export DATASET_ID=foam_slurm
export EXPR_ID=13204372

### August 7, 2023

Trying the pre-empt queue script from fresh terminal:

SETUP FOR CT_NVAE:
=============================
module load python
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export NERSC_GPU_ALLOCATION=m3562_g
export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
export MASTER_ADDR=$(hostname)
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
=============================

export DATASET_ID=foam_slurm
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=100000
export PNM=1e1

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM

Submitted batch job 13484185 - intervals in saving too large

Redo with smaller SAVE_INTERVAL:
export SAVE_INTERVAL=1000
sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM

Submitted batch job 13486270 - contrast low, ringing outside of bubble due to low reconstruction pnm

####  Redo with reconstruction pnm = creation pnm
export DATASET_ID=foam_slurm
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM

Submitted batch job 13489962 - input artifacts preserved

# Creation of datasets, 100 examples each

SETUP FOR DATASET CREATION:
=============================
module load python
export NERSC_GPU_ALLOCATION=m3562_g
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR
export NUM_EXAMPLES=100
=============================

## Synthetic Foam
export DATA_TYPE=foam
export IMAGE_ID=foam_100ex
export DATASET_ID=foam_45ang_100ex
export NUM_SPARSE_ANGLES=45
export RANDOM_ANGLES=True
export RING=0
export DO_PART_ONE=True
export DO_PART_TWO=False

sbatch --time=00:10:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $DO_PART_ONE $DO_PART_TWO

Submitted batch job 13491646 - SUCCESS


### Same images (wait until previous job is done), dataset WITHOUT a ring artifact
export RING=0
export DATASET_ID=foam_45ang_100ex
export DO_PART_ONE=False
export DO_PART_TWO=True
sbatch --time=00:02:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $DO_PART_ONE $DO_PART_TWO
Submitted batch job 13493342 - SUCCESS

### Same images, random is False:
export DATA_TYPE=foam
export IMAGE_ID=foam_100ex
export DATASET_ID=foam_45ang_100ex_uniform
export NUM_SPARSE_ANGLES=45
export RANDOM_ANGLES=False
export RING=0
export DO_PART_ONE=False
export DO_PART_TWO=True
sbatch --time=00:02:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $DO_PART_ONE $DO_PART_TWO
Submitted batch job 13500094 - SUCCESS

### Same images, but with a ring artifact
export RING=0.01
export DATASET_ID=foam_45ang_100ex_ring_0.01
export DO_PART_ONE=False
export DO_PART_TWO=True
sbatch --time=00:02:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $DO_PART_ONE $DO_PART_TWO
Submitted batch job 13494074 - SUCCESS

export RING=0.1
export DATASET_ID=foam_45ang_100ex_ring_0.1
export DO_PART_ONE=False
export DO_PART_TWO=True
sbatch --time=00:02:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $DO_PART_ONE $DO_PART_TWO
Submitted batch job 13494151 - SUCCESS


export RING=0.3
export DATASET_ID=foam_45ang_100ex_ring_0.3
export DO_PART_ONE=False
export DO_PART_TWO=True
sbatch --time=00:02:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $DO_PART_ONE $DO_PART_TWO
Submitted batch job 13495896 - SUCCESS

export RING=0.5
export DATASET_ID=foam_45ang_100ex_ring_0.5
export DO_PART_ONE=False
export DO_PART_TWO=True
sbatch --time=00:02:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $DO_PART_ONE $DO_PART_TWO
Submitted batch job 13494160 - ERROR
/pscratch/sd/v/vidyagan/CT_NVAE/computed_tomography/utils.py:85: RuntimeWarning: invalid value encountered in log
  sparse_sinogram = -np.log(sparse_sinogram_raw) # linearize the sinogram


## 3D COVID, contains a total of 650 3D examples
export DATA_TYPE=covid
export IMAGE_ID=covid_100ex
export DATASET_ID=covid_45ang_100ex
export COVID_RAW_DATA=/global/cfs/cdirs/m3562/users/hkim/real_data/raw

sbatch --time=00:10:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM $RING $DO_PART_ONE $DO_PART_TWO
Submitted batch job 13486562 - SUCCESS

## 3D Brain, contains a total of 25,000 3D examples

export DATA_TYPE=brain
export IMAGE_ID=brain_100ex
export DATASET_ID=brain_45ang_100ex

sbatch --time=00:10:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM $RING $DO_PART_ONE $DO_PART_TWO
Submitted batch job 13486606 - SUCCESS


# Submit jobs with pre-empt for brain and covid, 50 example, 90 angle datasets

## covid
module load python
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export NERSC_GPU_ALLOCATION=m3562_g
export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
export MASTER_ADDR=$(hostname)
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH

export DATASET_ID=covid_slurm
export BATCH_SIZE=1
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM
Submitted batch job 13498451 -- OOM

## brain

export DATASET_ID=brain_slurm
export BATCH_SIZE=1
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM
Submitted batch job 13498462 - input artifact preserved, low contrast is lost

### Submit pre-empt jobs for all the brain and covid 100 example, 45 angle datasets

# covid
export DATASET_ID=covid_45ang_100ex
export BATCH_SIZE=1
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM
Submitted batch job 13498758 - OOM

## brain
export DATASET_ID=brain_45ang_100ex
export BATCH_SIZE=1
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM
Submitted batch job 13498766 - input artifact preserved

# foam no artifact
export DATASET_ID=foam_45ang_100ex
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM
Submitted batch job 13498810 - blurry but reduced artifact

without modeling the artifact:
# foam ring 0.01
export DATASET_ID=foam_45ang_100ex_ring_0.01
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM
Submitted batch job 13498894 - strong artifact

# foam ring 0.1
export DATASET_ID=foam_45ang_100ex_ring_0.1
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM
Submitted batch job 13498904 - reconstruction basically a blank

# foam ring 0.3
export DATASET_ID=foam_45ang_100ex_ring_0.3
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM
Submitted batch job 13498914 - blurry mess but vague circular shape

WITH modeling the artifact: (ADDED THE $RING OPTION)
# foam ring 0.01
export RING=True
export DATASET_ID=foam_45ang_100ex_ring_0.01
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING
Submitted batch job 13502262 - blurry mess but vague circular shape

# foam ring 0.1
export RING=True
export DATASET_ID=foam_45ang_100ex_ring_0.1
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING
Submitted batch job 13502268 - a mess but not bad!

# foam ring 0.3
export RING=True
export DATASET_ID=foam_45ang_100ex_ring_0.3
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING
Submitted batch job 13502271 - a mess but some vague circular stuff

## Uniform
export RING=False
export DATASET_ID=foam_45ang_100ex_uniform
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING
Submitted batch job 13502275 - not great but resembles ground truth


# August 8, 2023

SETUP FOR DATASET CREATION:
=============================
module load python
export NERSC_GPU_ALLOCATION=m3562_g
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR
export NUM_EXAMPLES=100
=============================

Foam images in `images_foam_100ex` processed with sirt:
export DATA_TYPE=foam
export IMAGE_ID=foam_100ex
export DATASET_ID=foam_45ang_100ex_sirt
export NUM_SPARSE_ANGLES=45
export RANDOM_ANGLES=True
export RING=0
export ALGORITHM=sirt
export DO_PART_ONE=False
export DO_PART_TWO=True

sbatch --time=00:10:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO
Submitted batch job 13601242 - SUCCESS

Foam images in `images_foam_100ex` processed with tv:
export DATASET_ID=foam_45ang_100ex_tv
export ALGORITHM=tv

sbatch --time=00:10:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO
Submitted batch job 13601282 - SUCCESS

Covid images in `images_covid_100ex` processed with sirt:
export DATA_TYPE=covid
export IMAGE_ID=covid_100ex
export DATASET_ID=covid_45ang_100ex_sirt
export COVID_RAW_DATA=/global/cfs/cdirs/m3562/users/hkim/real_data/raw
export ALGORITHM=sirt

sbatch --time=01:00:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO
Submitted batch job 13602763 - SUCCESS

Covid images in `images_covid_100ex` processed with tv:
export DATA_TYPE=covid
export IMAGE_ID=covid_100ex
export DATASET_ID=covid_45ang_100ex_tv
export COVID_RAW_DATA=/global/cfs/cdirs/m3562/users/hkim/real_data/raw
export ALGORITHM=tv

sbatch --time=01:00:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO
Submitted batch job 13602780 - SUCCESS

Brain images in `images_brain_100ex` processed with sirt:
export DATA_TYPE=brain
export IMAGE_ID=brain_100ex
export DATASET_ID=brain_45ang_100ex_sirt
export ALGORITHM=sirt

sbatch --time=01:00:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO
Submitted batch job 13602838 - SUCCESS

Brain images in `images_brain_100ex` processed with tv:
export DATA_TYPE=brain
export IMAGE_ID=brain_100ex
export DATASET_ID=brain_45ang_100ex_tv
export ALGORITHM=tv

sbatch --time=01:00:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO
Submitted batch job 13602843 - SUCCESS


## Submit CT_NVAE preempt jobs

SETUP
=============================
module load python
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export NERSC_GPU_ALLOCATION=m3562_g
export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
export MASTER_ADDR=$(hostname)
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
=============================

## Foam
export RING=False
export DATASET_ID=foam_45ang_100ex_sirt
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING
Submitted batch job 13614584 - less artifact but blurry

export DATASET_ID=foam_45ang_100ex_tv
sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING
Submitted batch job 13614602 - heavy artifact

## Covid
export RING=False
export DATASET_ID=covid_45ang_100ex_sirt
export BATCH_SIZE=1
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING
Submitted batch job 13614647 - OOM

export DATASET_ID=covid_45ang_100ex_tv
sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING
Submitted batch job 13614650 - OOM

## Brain
export RING=False
export DATASET_ID=brain_45ang_100ex_sirt
export BATCH_SIZE=1
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING
Submitted batch job 13614653 - reconstructions appear to be the wrong image

export DATASET_ID=brain_45ang_100ex_tv
sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING
Submitted batch job 13614655 - not bad

## August 9, 2023
# Submit CT_NVAE jobs

SETUP
=============================
module load python
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export NERSC_GPU_ALLOCATION=m3562_g
export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
export MASTER_ADDR=$(hostname)
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
=============================

## Foam
export RING=False
export DATASET_ID=foam_45ang_100ex_tv
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING

Submitted batch job 13666938 - Pin memory thread exited unexpectedly

Try again 5 times:

Submitted batch job 13721579
Submitted batch job 13721581
Submitted batch job 13721595
Submitted batch job 13721596
Submitted batch job 13721598

Change to:
export DATASET_ID=foam_45ang_100ex
sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING

Try 5 times:
Submitted batch job 13721636
Submitted batch job 13721639
Submitted batch job 13721640 - preempt worked once
Submitted batch job 13721641
Submitted batch job 13721643 - restarting worked! however the checkpoint only saved at epoch 0


Changed to attempt to gracefully exit, trying 5 times:
export DATASET_ID=foam_45ang_100ex_tv
sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING

Submitted batch job 13737947
Submitted batch job 13737952
Submitted batch job 13737957
Submitted batch job 13737960
Submitted batch job 13737963

export PNM=1e1

Submitted batch job 13740713
Submitted batch job 13740717
Submitted batch job 13740728
Submitted batch job 13740731
Submitted batch job 13740732

export PNM=1e2

Submitted batch job 13740749
Submitted batch job 13740751
Submitted batch job 13740754
Submitted batch job 13740756
Submitted batch job 13740757

10 example COVID:
SETUP FOR DATASET CREATION:
=============================
module load python
export NERSC_GPU_ALLOCATION=m3562_g
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR
export NUM_EXAMPLES=10
=============================


export NUM_SPARSE_ANGLES=45
export RANDOM_ANGLES=True
export RING=0
export ALGORITHM=tv
export DO_PART_ONE=True
export DO_PART_TWO=True
export DATA_TYPE=covid
export IMAGE_ID=covid_10ex
export DATASET_ID=covid_45ang_10ex_tv
export COVID_RAW_DATA=/global/cfs/cdirs/m3562/users/hkim/real_data/raw

sbatch --time=00:10:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO
Submitted batch job 13738130



SETUP
=============================
module load python
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export NERSC_GPU_ALLOCATION=m3562_g
export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
export MASTER_ADDR=$(hostname)
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
=============================

export RING=False
export DATASET_ID=covid_45ang_10ex_tv
export BATCH_SIZE=1
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION --comment 96:00:00 $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING

Submitted 5 jobs:
Submitted batch job 13740664
Submitted batch job 13740667
Submitted batch job 13740668
Submitted batch job 13740670
Submitted batch job 13740672

## August 11, 2023
Try preempt script with smaller time:

Submitted batch job 13765626

## August 14, 2023

Figuring out preempt:


module load python
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR

sbatch $CT_NVAE_PATH/experimental/signal_interrupt.sh 

Submitted batch job 13890543
Submitted batch job 13890797

Removed #SBATCH --signal=SIGINT@60
Submitted batch job 13890843

Replaced #SBATCH --signal=SIGINT@60
13890898

Had to add "sys.stdout.flush()" to get output to print to file
Submitted batch job 13891027

NERSC example: https://gitlab.com/NERSC/checkpoint-on-signal-example

switched from python to exec: didn't work
switched to USR1 signal: didn't work

filed a helpdesk ticket at NERSC

## test_large_tomopy_reconstruction.py

SETUP:
================================
module load python
export NERSC_GPU_ALLOCATION=m3562_g
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
================================

python $CT_NVAE_PATH/computed_tomography/tests/test_large_tomopy_reconstruction.py


Testing signal interrupt with the preempt queue, 10 hour job:
module load python
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR

sbatch $CT_NVAE_PATH/experimental/signal_interrupt.sh 

Submitted batch job 13896375

# Testing CT_NVAE with preempt
Switched to srun

SETUP
=============================
module load python
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export NERSC_GPU_ALLOCATION=m3562_g
export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
export MASTER_ADDR=$(hostname)
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
=============================

## Foam
export RING=False
export DATASET_ID=foam_45ang_100ex_tv
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e2

sbatch -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING

Submitted batch job 13896436 - images don't match input and output
Submitted batch job 13896459
Submitted batch job 13896465 - images don't match
Submitted batch job 13896486
Submitted batch job 13896490

export CHECKPOINT_DIR=checkpts
export EXPR_ID=13896436
tensorboard --logdir $CHECKPOINT_DIR/eval-$EXPR_ID/

sbatch -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING

Submitted batch job 13896931

export BATCH_SIZE=16
sbatch -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING
Submitted batch job 13897171

export BATCH_SIZE=32
sbatch -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING
Submitted batch job 13897179 - Out of memory

PNM annealing:
export PNM=1e3
export BATCH_SIZE=8

sbatch -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING

Submitted batch job 13897336
Submitted batch job 13897338
Submitted batch job 13897342


export BATCH_SIZE=16
sbatch -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/train_single_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING
Submitted batch job 13897505

# Testing create_dataset_h5.py

SETUP
=============================
module load python
export NERSC_GPU_ALLOCATION=m3562_g
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR
export NUM_EXAMPLES=100
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
=============================

Foam images in `images_foam_100ex` processed with gridrec:
export DATA_TYPE=foam
export IMAGE_ID=foam_100ex
export DATASET_ID=foam_45ang_100ex_h5
export NUM_SPARSE_ANGLES=45
export RANDOM_ANGLES=True
export RING=0
export ALGORITHM=gridrec


python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_$IMAGE_ID --dest dataset_$DATASET_ID --train 0.7 --valid 0.2 --test 0.1 -n $NUM_EXAMPLES

python $CT_NVAE_PATH/preprocessing/create_dataset_h5.py --dir dataset_$DATASET_ID --sparse $NUM_SPARSE_ANGLES --random $RANDOM_ANGLES --ring $RING --pnm 1e3 --algorithm $ALGORITHM

## August 15, 2023

SETUP
=============================
module load python
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export NERSC_GPU_ALLOCATION=m3562_g
export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
export MASTER_ADDR=$(hostname)
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
=============================

## Foam
export RING=False
export DATASET_ID=foam_45ang_100ex_tv
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3



128 cpus:
sbatch -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/train_single_node_preempt_change_cpu.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING

all getting lots of iters
13945276 - gets worse at end
13945288
13945293
13945294
13945296

1 cpu:
13945314
13945316
13945319
13945320
13945321

Interactive node: For some reason can't get srun to work on interactive node

Try 2 nodes:

sbatch -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING

13945406
13945407
13945414
13945416
13945418
13945420

Changed node_rank
sbatch -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING

13946269
13946270
13946271
13946277
13946278

Looks like multinode training is working! (note: not really, see August 16)

Fixed cpus_per_task and increased to 4 nodes:
13946742
13946744
13946747
13946749
13946751

# August 16, 2023

added --num_proc_node=4 to train_multi_node_preempt.sh
NOTE: 3 places to change number of nodes, automate this


SETUP
=============================
module load python
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export NERSC_GPU_ALLOCATION=m3562_g
=============================

## Foam
export RING=False
export DATASET_ID=foam_45ang_100ex
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3
export NUM_NODES=4
export USE_H5=False

sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5

Submitted batch job 13973637
Submitted batch job 13973640
Submitted batch job 13973641
Submitted batch job 13973642
Submitted batch job 13973643

# August 17, 2023

wandb error, trying the above again:

14003850
14003852
14003855
14003856
14003857

error again with parallelization, trying again:
14006332
14006336
14006338
14006342
14006344

rolling back
14006725


moving master addr into the script
14006842

commented out MASTER_PORT
14007246


removed master_addr from inside script
14007294

Switch workflow to specifying the number of nodes (1 task per node):

SETUP
=============================
module load python
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export NERSC_GPU_ALLOCATION=m3562_g
export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
export MASTER_ADDR=$(hostname)
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
export NUM_NODES=1
=============================

## Foam
export RING=False
export DATASET_ID=foam_45ang_100ex
export USE_H5=False
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5

export NUM_NODES=1
Submitted batch job 14003960
Submitted batch job 14003961
Submitted batch job 14003963
Submitted batch job 14003964
Submitted batch job 14003966

Remake foam h5 dataset:

SETUP
=============================
module load python
export NERSC_GPU_ALLOCATION=m3562_g
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR
export NUM_EXAMPLES=100
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
=============================

Foam images in `images_foam_100ex` processed with gridrec:
export DATA_TYPE=foam
export IMAGE_ID=foam_100ex
export DATASET_ID=foam_45ang_100ex_h5
export NUM_SPARSE_ANGLES=45
export RANDOM_ANGLES=True
export RING=0
export ALGORITHM=gridrec


python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_$IMAGE_ID --dest dataset_$DATASET_ID --train 0.7 --valid 0.2 --test 0.1 -n $NUM_EXAMPLES

python $CT_NVAE_PATH/preprocessing/create_dataset_h5.py --dir dataset_$DATASET_ID --sparse $NUM_SPARSE_ANGLES --random $RANDOM_ANGLES --ring $RING --pnm 1e3 --algorithm $ALGORITHM


salloc -N 1 --time=120 -C gpu -A $NERSC_GPU_ALLOCATION --qos=interactive --cpus-per-task 128


## Using H5 loading

SETUP
=============================
module load python
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export NERSC_GPU_ALLOCATION=m3562_g
export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
export MASTER_ADDR=$(hostname)
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
export NUM_NODES=1
=============================

## Foam
export RING=False
export DATASET_ID=foam_45ang_100ex_h5
export USE_H5=True
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3

sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5

# Do a Timing Comparison using H5 and using numpy
h5 is about the same

# working on getting multinode training to work

SETUP
=============================
module load python
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export NERSC_GPU_ALLOCATION=m3562_g
=============================

## Foam
export RING=False
export DATASET_ID=foam_45ang_100ex_h5
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3
export NUM_NODES=1
export USE_H5=True

sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5

Submitted batch job 14013375 (canceled)

export NUM_NODES=2
sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5

14013390

Fixed pnm anneal:
export NUM_NODES=1
sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5

14014219
14014227
14014232
14014238
14014242

2 nodes:
14014711

export BATCH_SIZE=16
export NUM_NODES=1
sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5
Submitted batch job 14014929

export BATCH_SIZE=16
export NUM_NODES=2
sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5
Submitted batch job 14014939

export BATCH_SIZE=16
export NUM_NODES=3
sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5
Submitted batch job 14014955 - looks good except still has speckle artifact

export BATCH_SIZE=16
export NUM_NODES=4
sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5
Submitted batch job 14014965

changed number of warmup epochs to 400:
export BATCH_SIZE=16
export NUM_NODES=4
sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5
14015164

More nodes:
export BATCH_SIZE=16
export NUM_NODES=16
sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5
Submitted batch job 14019884 - completely blank result, try again


# August 18, 2023

Fixed boolean command line args:

SETUP
=============================
module load python
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export NERSC_GPU_ALLOCATION=m3562_g
=============================

## Foam
export RING=False
export DATASET_ID=foam_45ang_100ex_h5
export BATCH_SIZE=8
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3
export NUM_NODES=3
export USE_H5=True

sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5

Submitted batch job 14049425 - looks good except still has speckle artifact
Submitted batch job 14049427 - looks good, a little blurring, still has speckle artifact
Submitted batch job 14049430 - same as above

# Full Dataset Creation with H5

SETUP
=============================
module load python
export NERSC_GPU_ALLOCATION=m3562_g
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR
export NUM_EXAMPLES=1000
=============================

export NUM_SPARSE_ANGLES=10
export RANDOM_ANGLES=True
export RING=0
export ALGORITHM=gridrec
export DO_PART_ONE=True
export DO_PART_TWO=True
export DATA_TYPE=foam
export IMAGE_ID=foam_1000ex
export DATASET_ID=foam_10ang_1000ex

sbatch --time=01:00:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO

Submitted batch job 14051567


# Debugging preempt

sbatch $CT_NVAE_PATH/experimental/signal_interrupt.sh
Submitted batch job 14051613


Sweep number of angles script:

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets.sh



RUN ME:

Training on the dataset sweep
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train.sh

Ran 2x:

(actually training below, there is a typo)
Current NUM_SPARSE_ANGLES: 10
Submitting job to create foam_10ang_1000ex
Submitted batch job 14060762
Current NUM_SPARSE_ANGLES: 20
Submitting job to create foam_20ang_1000ex
Submitted batch job 14060770
Current NUM_SPARSE_ANGLES: 30
Submitting job to create foam_30ang_1000ex
Submitted batch job 14060772
Current NUM_SPARSE_ANGLES: 40
Submitting job to create foam_40ang_1000ex
Submitted batch job 14060775
Current NUM_SPARSE_ANGLES: 50
Submitting job to create foam_50ang_1000ex
Submitted batch job 14060777
Current NUM_SPARSE_ANGLES: 60
Submitting job to create foam_60ang_1000ex
Submitted batch job 14060778
Current NUM_SPARSE_ANGLES: 70
Submitting job to create foam_70ang_1000ex
Submitted batch job 14060779
Current NUM_SPARSE_ANGLES: 80
Submitting job to create foam_80ang_1000ex
Submitted batch job 14060780
Current NUM_SPARSE_ANGLES: 90
Submitting job to create foam_90ang_1000ex
Submitted batch job 14060781
Current NUM_SPARSE_ANGLES: 100
Submitting job to create foam_100ang_1000ex
Submitted batch job 14060782 -- doesn't look like it has the speckle artifact
Current NUM_SPARSE_ANGLES: 110
Submitting job to create foam_110ang_1000ex
Submitted batch job 14060794 - back to speckle artifact
Current NUM_SPARSE_ANGLES: 120
Submitting job to create foam_120ang_1000ex
Submitted batch job 14060795
Current NUM_SPARSE_ANGLES: 130
Submitting job to create foam_130ang_1000ex
Submitted batch job 14060811
Current NUM_SPARSE_ANGLES: 140
Submitting job to create foam_140ang_1000ex
Submitted batch job 14060812
Current NUM_SPARSE_ANGLES: 150
Submitting job to create foam_150ang_1000ex
Submitted batch job 14060813
Current NUM_SPARSE_ANGLES: 160
Submitting job to create foam_160ang_1000ex
Submitted batch job 14060815
Current NUM_SPARSE_ANGLES: 170
Submitting job to create foam_170ang_1000ex
Submitted batch job 14060816
Current NUM_SPARSE_ANGLES: 180
Submitting job to create foam_180ang_1000ex
Submitted batch job 14060817


====

Current NUM_SPARSE_ANGLES: 10
Submitting job to train with foam_10ang_1000ex
Submitted batch job 14060826
Current NUM_SPARSE_ANGLES: 20
Submitting job to train with foam_20ang_1000ex
Submitted batch job 14060827
Current NUM_SPARSE_ANGLES: 30
Submitting job to train with foam_30ang_1000ex
Submitted batch job 14060828
Current NUM_SPARSE_ANGLES: 40
Submitting job to train with foam_40ang_1000ex
Submitted batch job 14060829 - In the middle, this is an almost perfect reconstruction
Current NUM_SPARSE_ANGLES: 50
Submitting job to train with foam_50ang_1000ex
Submitted batch job 14060830
Current NUM_SPARSE_ANGLES: 60
Submitting job to train with foam_60ang_1000ex
Submitted batch job 14060831
Current NUM_SPARSE_ANGLES: 70
Submitting job to train with foam_70ang_1000ex
Submitted batch job 14060833
Current NUM_SPARSE_ANGLES: 80
Submitting job to train with foam_80ang_1000ex
Submitted batch job 14060834
Current NUM_SPARSE_ANGLES: 90
Submitting job to train with foam_90ang_1000ex
Submitted batch job 14060838 - In the middle, this is an almost perfect reconstruction
Current NUM_SPARSE_ANGLES: 100
Submitting job to train with foam_100ang_1000ex
Submitted batch job 14060839
Current NUM_SPARSE_ANGLES: 110
Submitting job to train with foam_110ang_1000ex
Submitted batch job 14060840
Current NUM_SPARSE_ANGLES: 120
Submitting job to train with foam_120ang_1000ex
Submitted batch job 14060843
Current NUM_SPARSE_ANGLES: 130
Submitting job to train with foam_130ang_1000ex
Submitted batch job 14060845
Current NUM_SPARSE_ANGLES: 140
Submitting job to train with foam_140ang_1000ex
Submitted batch job 14060846
Current NUM_SPARSE_ANGLES: 150
Submitting job to train with foam_150ang_1000ex
Submitted batch job 14060848
Current NUM_SPARSE_ANGLES: 160
Submitting job to train with foam_160ang_1000ex
Submitted batch job 14060849
Current NUM_SPARSE_ANGLES: 170
Submitting job to train with foam_170ang_1000ex
Submitted batch job 14060850
Current NUM_SPARSE_ANGLES: 180
Submitting job to train with foam_180ang_1000ex
Submitted batch job 14060851

How to count all jobs running and pending:
squeue -u vidyagan -h -t pending,running -r | wc -l

# August 21, 2023

When the algorithm is doing well, suddenly the figure of merit gets terrible, changing sign
What are possible causes of this instability?
Final distribution too peaky
log or sqrt operations
Can increase regularization
annealing of pnm -- turn off the annealing and see if it still happens
catch when this happens, roll back to the last checkpoint, add a tiny bit of noise and continue training

Noise artifact:

Is there too much noise in the initial measurements?
Might want to reduce noise to see the different between low dose, many measurements, and high dose, few measurements
understand the halo


Redo the sweep, saving the model at the best iteration:

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train.sh

Current NUM_SPARSE_ANGLES: 10
Submitting job to train with foam_10ang_10e00ex
Submitted batch job 14203298
Current NUM_SPARSE_ANGLES: 20
Submitting job to train with foam_20ang_1000ex
Submitted batch job 14203301
Current NUM_SPARSE_ANGLES: 30
Submitting job to train with foam_30ang_1000ex
Submitted batch job 14203303
Current NUM_SPARSE_ANGLES: 40
Submitting job to train with foam_40ang_1000ex
Submitted batch job 14203305
Current NUM_SPARSE_ANGLES: 50
Submitting job to train with foam_50ang_1000ex
Submitted batch job 14203307
Current NUM_SPARSE_ANGLES: 60
Submitting job to train with foam_60ang_1000ex
Submitted batch job 14203309
Current NUM_SPARSE_ANGLES: 70
Submitting job to train with foam_70ang_1000ex
Submitted batch job 14203310
Current NUM_SPARSE_ANGLES: 80
Submitting job to train with foam_80ang_1000ex
Submitted batch job 14203311
Current NUM_SPARSE_ANGLES: 90
Submitting job to train with foam_90ang_1000ex
Submitted batch job 14203312
Current NUM_SPARSE_ANGLES: 100
Submitting job to train with foam_100ang_1000ex
Submitted batch job 14203313
Current NUM_SPARSE_ANGLES: 110
Submitting job to train with foam_110ang_1000ex
Submitted batch job 14203314
Current NUM_SPARSE_ANGLES: 120
Submitting job to train with foam_120ang_1000ex
Submitted batch job 14203315
Current NUM_SPARSE_ANGLES: 130
Submitting job to train with foam_130ang_1000ex
Submitted batch job 14203316
Current NUM_SPARSE_ANGLES: 140
Submitting job to train with foam_140ang_1000ex
Submitted batch job 14203317
Current NUM_SPARSE_ANGLES: 150
Submitting job to train with foam_150ang_1000ex
Submitted batch job 14203318
Current NUM_SPARSE_ANGLES: 160
Submitting job to train with foam_160ang_1000ex
Submitted batch job 14203319
Current NUM_SPARSE_ANGLES: 170
Submitting job to train with foam_170ang_1000ex
Submitted batch job 14203323
Current NUM_SPARSE_ANGLES: 180
Submitting job to train with foam_180ang_1000ex
Submitted batch job 14203325

# August 22, 2023
cancel all jobs from a user
scancel -u <username>

Pipe script to an output log:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train.sh >> output_aug_22_2023.txt
Canceled after submitting 1 copy of each job


Create the original covid image set:

SETUP
=============================
module load python
export NERSC_GPU_ALLOCATION=m3562_g
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR
export NUM_EXAMPLES=650
=============================

export NUM_SPARSE_ANGLES=10
export RANDOM_ANGLES=True
export RING=0
export ALGORITHM=gridrec
export DO_PART_ONE=True
export DO_PART_TWO=False
export DATA_TYPE=covid
export IMAGE_ID=covid_650ex
export DATASET_ID=covid_10ang_650ex

sbatch --time=02:00:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO

Submitted batch job 14248647
FINISHED

Making covid datasets: DONE
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets.sh

Current NUM_SPARSE_ANGLES: 10
Submitting job to create covid_10ang_1000ex
Submitted batch job 14262363
Current NUM_SPARSE_ANGLES: 20
Submitting job to create covid_20ang_1000ex
Submitted batch job 14262365
Current NUM_SPARSE_ANGLES: 30
Submitting job to create covid_30ang_1000ex
Submitted batch job 14262367
Current NUM_SPARSE_ANGLES: 40
Submitting job to create covid_40ang_1000ex
Submitted batch job 14262371
Current NUM_SPARSE_ANGLES: 50
Submitting job to create covid_50ang_1000ex
Submitted batch job 14262375
Current NUM_SPARSE_ANGLES: 60
Submitting job to create covid_60ang_1000ex
Submitted batch job 14262377
Current NUM_SPARSE_ANGLES: 70
Submitting job to create covid_70ang_1000ex
Submitted batch job 14262379
Current NUM_SPARSE_ANGLES: 80
Submitting job to create covid_80ang_1000ex
Submitted batch job 14262381
Current NUM_SPARSE_ANGLES: 90
Submitting job to create covid_90ang_1000ex
Submitted batch job 14262384
Current NUM_SPARSE_ANGLES: 100
Submitting job to create covid_100ang_1000ex
Submitted batch job 14262387
Current NUM_SPARSE_ANGLES: 110
Submitting job to create covid_110ang_1000ex
Submitted batch job 14262391
Current NUM_SPARSE_ANGLES: 120
Submitting job to create covid_120ang_1000ex
Submitted batch job 14262397
Current NUM_SPARSE_ANGLES: 130
Submitting job to create covid_130ang_1000ex
Submitted batch job 14262398
Current NUM_SPARSE_ANGLES: 140
Submitting job to create covid_140ang_1000ex
Submitted batch job 14262399
Current NUM_SPARSE_ANGLES: 150
Submitting job to create covid_150ang_1000ex
Submitted batch job 14262400
Current NUM_SPARSE_ANGLES: 160
Submitting job to create covid_160ang_1000ex
Submitted batch job 14262401
Current NUM_SPARSE_ANGLES: 170
Submitting job to create covid_170ang_1000ex
Submitted batch job 14262402
Current NUM_SPARSE_ANGLES: 180
Submitting job to create covid_180ang_1000ex
Submitted batch job 14262404

Update foam training sweep script to have correct DATASET_ID.
Running:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train.sh >> output_aug_22_2023_v2.txt


TODO: Create the brain dataset

# August 23, 2023

remaking the covid dataset with 1000 examples, 10 angles and 20 angles

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets.sh

Current NUM_SPARSE_ANGLES: 10
Submitting job to create covid_10ang_1000ex
Submitted batch job 14301741 - DONE
Current NUM_SPARSE_ANGLES: 20
Submitting job to create covid_20ang_1000ex
Submitted batch job 14301743 - DONE

Renamed all the covid datasets to 650ex and removed 10,30,50,70,90,110,130,150,170 datasets

Training for the covid dataset:
on NoMachine:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train.sh >> output_aug_23_2023.txt
ERROR --> DEBUG ME

Create the brain images:
=============================
module load python
export NERSC_GPU_ALLOCATION=m3562_g
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR
export NUM_EXAMPLES=1000
=============================

export NUM_SPARSE_ANGLES=10
export RANDOM_ANGLES=True
export RING=0
export ALGORITHM=gridrec
export DO_PART_ONE=True
export DO_PART_TWO=False
export DATA_TYPE=brain
export IMAGE_ID=brain_1000ex
export DATASET_ID=brain_10ang_1000ex

sbatch --time=02:00:00 -A m3562_g $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO

Submitted batch job 14312064 - SUCCESS

Create the brain datasets:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets.sh
DONE

Current NUM_SPARSE_ANGLES: 20
Submitting job to create brain_20ang_1000ex
Submitted batch job 14313877
Current NUM_SPARSE_ANGLES: 40
Submitting job to create brain_40ang_1000ex
Submitted batch job 14313879
Current NUM_SPARSE_ANGLES: 60
Submitting job to create brain_60ang_1000ex
Submitted batch job 14313880
Current NUM_SPARSE_ANGLES: 80
Submitting job to create brain_80ang_1000ex
Submitted batch job 14313881
Current NUM_SPARSE_ANGLES: 100
Submitting job to create brain_100ang_1000ex
Submitted batch job 14313882
Current NUM_SPARSE_ANGLES: 120
Submitting job to create brain_120ang_1000ex
Submitted batch job 14313883
Current NUM_SPARSE_ANGLES: 140
Submitting job to create brain_140ang_1000ex
Submitted batch job 14313884
Current NUM_SPARSE_ANGLES: 160
Submitting job to create brain_160ang_1000ex
Submitted batch job 14313885
Current NUM_SPARSE_ANGLES: 180
Submitting job to create brain_180ang_1000ex
Submitted batch job 14313886


Create ring artifacts dataset:
For foam images:

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets.sh
DONE

Current NUM_SPARSE_ANGLES: 20
Submitting job to create foam_20ang_1000ex_0.01ring
Submitted batch job 14312980
Current NUM_SPARSE_ANGLES: 40
Submitting job to create foam_40ang_1000ex_0.01ring
Submitted batch job 14312981
Current NUM_SPARSE_ANGLES: 60
Submitting job to create foam_60ang_1000ex_0.01ring
Submitted batch job 14312982
Current NUM_SPARSE_ANGLES: 80
Submitting job to create foam_80ang_1000ex_0.01ring
Submitted batch job 14312983
Current NUM_SPARSE_ANGLES: 100
Submitting job to create foam_100ang_1000ex_0.01ring
Submitted batch job 14312985
Current NUM_SPARSE_ANGLES: 120
Submitting job to create foam_120ang_1000ex_0.01ring
Submitted batch job 14312986
Current NUM_SPARSE_ANGLES: 140
Submitting job to create foam_140ang_1000ex_0.01ring
Submitted batch job 14312988
Current NUM_SPARSE_ANGLES: 160
Submitting job to create foam_160ang_1000ex_0.01ring
Submitted batch job 14312989
Current NUM_SPARSE_ANGLES: 180
Submitting job to create foam_180ang_1000ex_0.01ring
Submitted batch job 14312990

# analysis of the results from the long foam sweep
Original job IDs below:

Current NUM_SPARSE_ANGLES: 10
Submitting job to train with foam_10ang_1000ex
Job ID: 14263357
Current NUM_SPARSE_ANGLES: 20
Submitting job to train with foam_20ang_1000ex
Job ID: 14263361
Current NUM_SPARSE_ANGLES: 30
Submitting job to train with foam_30ang_1000ex
Job ID: 14263362
Current NUM_SPARSE_ANGLES: 40
Submitting job to train with foam_40ang_1000ex
Job ID: 14263365
Current NUM_SPARSE_ANGLES: 50
Submitting job to train with foam_50ang_1000ex
Job ID: 14263367
Current NUM_SPARSE_ANGLES: 60
Submitting job to train with foam_60ang_1000ex
Job ID: 14263370
Current NUM_SPARSE_ANGLES: 70
Submitting job to train with foam_70ang_1000ex
Job ID: 14263372
Current NUM_SPARSE_ANGLES: 80
Submitting job to train with foam_80ang_1000ex
Job ID: 14263375
Current NUM_SPARSE_ANGLES: 90
Submitting job to train with foam_90ang_1000ex
Job ID: 14263377
Current NUM_SPARSE_ANGLES: 100
Submitting job to train with foam_100ang_1000ex
Job ID: 14263384
Current NUM_SPARSE_ANGLES: 110
Submitting job to train with foam_110ang_1000ex
Job ID: 14263388
Current NUM_SPARSE_ANGLES: 120
Submitting job to train with foam_120ang_1000ex
Job ID: 14263391
Current NUM_SPARSE_ANGLES: 130
Submitting job to train with foam_130ang_1000ex
Job ID: 14263396
Current NUM_SPARSE_ANGLES: 140
Submitting job to train with foam_140ang_1000ex
Job ID: 14263400
Current NUM_SPARSE_ANGLES: 150
Submitting job to train with foam_150ang_1000ex
Job ID: 14263402
Current NUM_SPARSE_ANGLES: 160
Submitting job to train with foam_160ang_1000ex
Job ID: 14263406
Current NUM_SPARSE_ANGLES: 170
Submitting job to train with foam_170ang_1000ex
Job ID: 14263410
Current NUM_SPARSE_ANGLES: 180
Submitting job to train with foam_180ang_1000ex
Job ID: 14263412



Workflow to analyze the results:


export DATASET_ID=foam_180ang_1000ex
export SAVE_NAME=14263412

conda deactivate
module purge
module load python
export NERSC_GPU_ALLOCATION=m3562_g
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR


export RING=False
export BATCH_SIZE=16
export EPOCHS=0
export SAVE_INTERVAL=1000
export PNM=1e3
export NUM_NODES=1
export NUM_GPUS=1
export SLURM_PROCID=0
export SLURM_JOB_ID=0
export SLURM_STEP_ID=0
export USE_H5=True

export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
export MASTER_ADDR=$(hostname)
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH

echo $MASTER_ADDR

# Parameters for the foam dataset
export NUM_LATENT_SCALES=2
export NUM_GROUPS_PER_SCALE=10
export NUM_POSTPROCESS_CELLS=3
export NUM_PREPROCESS_CELLS=3
export NUM_CELL_PER_COND_ENC=2
export NUM_CELL_PER_COND_DEC=2
export NUM_LATENT_PER_GROUP=20
export NUM_PREPROCESS_BLOCKS=2
export NUM_POSTPROCESS_BLOCKS=2
export WEIGHT_DECAY_NORM=1e-2
export NUM_CHANNELS_ENC=32
export NUM_CHANNELS_DEC=32
export NUM_NF=0
export MIN_GROUPS_PER_SCALE=1

python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $SAVE_NAME --dataset $DATASET_ID --batch_size $BATCH_SIZE --epochs $EPOCHS --num_latent_scales $NUM_LATENT_SCALES --num_groups_per_scale $NUM_GROUPS_PER_SCALE --num_postprocess_cells $NUM_POSTPROCESS_CELLS --num_preprocess_cells $NUM_PREPROCESS_CELLS --num_cell_per_cond_enc $NUM_CELL_PER_COND_ENC --num_cell_per_cond_dec $NUM_CELL_PER_COND_DEC --num_latent_per_group $NUM_LATENT_PER_GROUP --num_preprocess_blocks $NUM_PREPROCESS_BLOCKS --num_postprocess_blocks $NUM_POSTPROCESS_BLOCKS --weight_decay_norm $WEIGHT_DECAY_NORM --num_channels_enc $NUM_CHANNELS_ENC --num_channels_dec $NUM_CHANNELS_DEC --num_nf $NUM_NF  --ada_groups --num_process_per_node $NUM_GPUS --use_se --res_dist --fast_adamax --pnm $PNM --save_interval $SAVE_INTERVAL --cont_training --model_ring_artifact $RING --num_proc_node $NUM_NODES --use_h5 $USE_H5 --min_groups_per_scale $MIN_GROUPS_PER_SCALE --use_nersc


# August 24, 2023


# Interactive node command
salloc -N 1 -n 1 --time=120 -C gpu -A m3562_g --qos=interactive --cpus-per-task=128


# Starting back from the beginning on training foam dataset

training:

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_24_2023_train_1.txt

(output file saved in home directory)
cd
cat output_aug_24_2023_train_1.txt

FAILED

REDO:
training:
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_24_2023_train_2.txt
STOPPED
14362251 14362253 14362255 14362257 14362258 14362260 14362264 14362267 14362269 14362270 14362271 14362274 14362279 14362283 14362285 14362287 14362288 14362290



Evaluate foam training runs:
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_24_2023_evaluate.txt

Analyze training results:

module load python
conda activate tomopy

export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
cd $WORKING_DIR

export EXPR_ID=14263391
export EPOCH=560
export RANK=0


python $CT_NVAE_PATH/metrics/analyze_training_results.py --checkpoint_dir $CHECKPOINT_DIR --expr_id $EXPR_ID --rank $RANK --original_size 128 --dataset_type valid --epoch $EPOCH

Path to H5 file:
/pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-14263391/eval_dataset_valid_epoch_560_rank_5.h5

(Something is buggy about the procedure, but sort of working)

REDO of full training run:
training:
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_24_2023_train_3.txt
STOPPED

Going really slow, removed --use_nersc
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_24_2023_train_4.txt
Still going slow

Re-try again:
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_24_2023_train_7.txt

Commented out validation step:

export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_24_2023_train_8.txt
COMPLETED

# August 25, 2023

Switch to allocation m2859_g

export NERSC_GPU_ALLOCATION=m2859_g

evaluation run:
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_24_2023_train_9.txt

Ran again with train and valid set:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_24_2023_train_11.txt

Analyze training results:

module load python
conda activate tomopy

export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
cd $WORKING_DIR

export EXPR_ID=14374880
export EPOCH=351
export RANK=0
export DATASET_TYPE=valid

python $CT_NVAE_PATH/metrics/analyze_training_results.py --checkpoint_dir $CHECKPOINT_DIR --expr_id $EXPR_ID --rank $RANK --original_size 128 --dataset_type $DATASET_TYPE --epoch $EPOCH

Path to H5 file:
/pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-14374866/eval_dataset_train_epoch_213_rank_0.h5

Problem in visualizing: went away when switched to model.train() in the evaluation step

Algorithm failed to do well:
14374876


Redo-ing to have splits done only once: 
conda activate tomopy
export IMAGES_DIR=images_foam_1000ex
export NUM_IMG=1000
cd $WORKING_DIR

python $CT_NVAE_PATH/preprocessing/create_splits.py --src $IMAGES_DIR --dest $IMAGES_DIR --train 0.7 --valid 0.2 --test 0.1 -n $NUM_IMG




Change me: create_dataset.sh, fix the dataset creation sweep - CHANGED

For foam images:
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh

Current NUM_SPARSE_ANGLES: 20
Submitting job to create foam_20ang_1000ex_0ring
Submitted batch job 14409629
Current NUM_SPARSE_ANGLES: 40
Submitting job to create foam_40ang_1000ex_0ring
Submitted batch job 14409635
Current NUM_SPARSE_ANGLES: 60
Submitting job to create foam_60ang_1000ex_0ring
Submitted batch job 14409639
Current NUM_SPARSE_ANGLES: 80
Submitting job to create foam_80ang_1000ex_0ring
Submitted batch job 14409641
Current NUM_SPARSE_ANGLES: 100
Submitting job to create foam_100ang_1000ex_0ring
Submitted batch job 14409642
Current NUM_SPARSE_ANGLES: 120
Submitting job to create foam_120ang_1000ex_0ring
Submitted batch job 14409645
Current NUM_SPARSE_ANGLES: 140
Submitting job to create foam_140ang_1000ex_0ring
Submitted batch job 14409648
Current NUM_SPARSE_ANGLES: 160
Submitting job to create foam_160ang_1000ex_0ring
Submitted batch job 14409649
Current NUM_SPARSE_ANGLES: 180
Submitting job to create foam_180ang_1000ex_0ring
Submitted batch job 14409651


For foam ring:
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam_ring.sh

Current NUM_SPARSE_ANGLES: 20
Submitting job to create foam_20ang_1000ex_0.01ring
Submitted batch job 14415675
Current NUM_SPARSE_ANGLES: 40
Submitting job to create foam_40ang_1000ex_0.01ring
Submitted batch job 14415677
Current NUM_SPARSE_ANGLES: 60
Submitting job to create foam_60ang_1000ex_0.01ring
Submitted batch job 14415678
Current NUM_SPARSE_ANGLES: 80
Submitting job to create foam_80ang_1000ex_0.01ring
Submitted batch job 14415679
Current NUM_SPARSE_ANGLES: 100
Submitting job to create foam_100ang_1000ex_0.01ring
Submitted batch job 14415680
Current NUM_SPARSE_ANGLES: 120
Submitting job to create foam_120ang_1000ex_0.01ring
Submitted batch job 14415681
Current NUM_SPARSE_ANGLES: 140
Submitting job to create foam_140ang_1000ex_0.01ring
Submitted batch job 14415682
Current NUM_SPARSE_ANGLES: 160
Submitting job to create foam_160ang_1000ex_0.01ring
Submitted batch job 14415683
Current NUM_SPARSE_ANGLES: 180
Submitting job to create foam_180ang_1000ex_0.01ring
Submitted batch job 14415684

For foam ring 2:
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam_ring_2.sh
Current NUM_SPARSE_ANGLES: 20
Submitting job to create foam_20ang_1000ex_0.1ring
Submitted batch job 14421602
Current NUM_SPARSE_ANGLES: 40
Submitting job to create foam_40ang_1000ex_0.1ring
Submitted batch job 14421604
Current NUM_SPARSE_ANGLES: 60
Submitting job to create foam_60ang_1000ex_0.1ring
Submitted batch job 14421605
Current NUM_SPARSE_ANGLES: 80
Submitting job to create foam_80ang_1000ex_0.1ring
Submitted batch job 14421606
Current NUM_SPARSE_ANGLES: 100
Submitting job to create foam_100ang_1000ex_0.1ring
Submitted batch job 14421607
Current NUM_SPARSE_ANGLES: 120
Submitting job to create foam_120ang_1000ex_0.1ring
Submitted batch job 14421608
Current NUM_SPARSE_ANGLES: 140
Submitting job to create foam_140ang_1000ex_0.1ring
Submitted batch job 14421610
Current NUM_SPARSE_ANGLES: 160
Submitting job to create foam_160ang_1000ex_0.1ring
Submitted batch job 14421611
Current NUM_SPARSE_ANGLES: 180
Submitting job to create foam_180ang_1000ex_0.1ring
Submitted batch job 14421612

For covid:
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_covid.sh

Current NUM_SPARSE_ANGLES: 20
Submitting job to create covid_20ang_650ex_0ring
Submitted batch job 14415751
Current NUM_SPARSE_ANGLES: 40
Submitting job to create covid_40ang_650ex_0ring
Submitted batch job 14415752
Current NUM_SPARSE_ANGLES: 60
Submitting job to create covid_60ang_650ex_0ring
Submitted batch job 14415753
Current NUM_SPARSE_ANGLES: 80
Submitting job to create covid_80ang_650ex_0ring
Submitted batch job 14415754
Current NUM_SPARSE_ANGLES: 100
Submitting job to create covid_100ang_650ex_0ring
Submitted batch job 14415755
Current NUM_SPARSE_ANGLES: 120
Submitting job to create covid_120ang_650ex_0ring
Submitted batch job 14415756
Current NUM_SPARSE_ANGLES: 140
Submitting job to create covid_140ang_650ex_0ring
Submitted batch job 14415757
Current NUM_SPARSE_ANGLES: 160
Submitting job to create covid_160ang_650ex_0ring
Submitted batch job 14415758
Current NUM_SPARSE_ANGLES: 180
Submitting job to create covid_180ang_650ex_0ring
Submitted batch job 14415759

For brain:
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_brain.sh

Current NUM_SPARSE_ANGLES: 20
Submitting job to create brain_20ang_1000ex_0ring
Submitted batch job 14415802
Current NUM_SPARSE_ANGLES: 40
Submitting job to create brain_40ang_1000ex_0ring
Submitted batch job 14415803
Current NUM_SPARSE_ANGLES: 60
Submitting job to create brain_60ang_1000ex_0ring
Submitted batch job 14415804
Current NUM_SPARSE_ANGLES: 80
Submitting job to create brain_80ang_1000ex_0ring
Submitted batch job 14415805
Current NUM_SPARSE_ANGLES: 100
Submitting job to create brain_100ang_1000ex_0ring
Submitted batch job 14415806
Current NUM_SPARSE_ANGLES: 120
Submitting job to create brain_120ang_1000ex_0ring
Submitted batch job 14415807
Current NUM_SPARSE_ANGLES: 140
Submitting job to create brain_140ang_1000ex_0ring
Submitted batch job 14415808
Current NUM_SPARSE_ANGLES: 160
Submitting job to create brain_160ang_1000ex_0ring
Submitted batch job 14415810
Current NUM_SPARSE_ANGLES: 180
Submitting job to create brain_180ang_1000ex_0ring
Submitted batch job 14415811


sweep_num_proj_train_foam.sh --> sweep_num_proj_train_foam_0.sh

FOAM JOBS START ##############################################################################################################
New set of foam jobs:
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_25_2023_train_1.txt
14416819 14416820 14416821 14416822 14416823 14416824 14416825 14416826 14416827
EVAL:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_25_2023_train_1_eval.txt
Analyzing results in a sweep:
export JOB_ID_ARRAY=(14416819 14416820 14416821 14416822 14416823 14416824 14416825 14416826 14416827)
export INPUT_FILE="analyze_sweep_input_1.txt"
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/analyze_sweep.sh $JOB_ID_ARRAY $INPUT_FILE
Submitted batch job 14597701

Foam ring jobs (with removal of ring artifact):
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_ring.sh >> output_aug_25_2023_train_2.txt
14418069 14418070 14418071 14418073 14418074 14418075 14418077 14418079 14418080
EVAL:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_ring.sh >> output_aug_25_2023_train_2_eval.txt
Analyzing results in a sweep:
export JOB_ID_ARRAY=(14418069 14418070 14418071 14418073 14418074 14418075 14418077 14418079 14418080)
export INPUT_FILE="analyze_sweep_input_2.txt"
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/analyze_sweep.sh $JOB_ID_ARRAY $INPUT_FILE
Submitted batch job 14597703

Foam ring jobs (without removal of ring artifact):
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_ring_2.sh >> output_aug_25_2023_train_3.txt
14418293 14418295 14418296 14418298 14418299 14418301 14418302 14418303 14418305
EVAL:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_ring_2.sh >> output_aug_25_2023_train_3_eval.txt
Analyzing results in a sweep:
export JOB_ID_ARRAY=(14418293 14418295 14418296 14418298 14418299 14418301 14418302 14418303 14418305)
export INPUT_FILE="analyze_sweep_input_3.txt"
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/analyze_sweep.sh $JOB_ID_ARRAY $INPUT_FILE
Submitted batch job 14597705

Foam ring jobs (with removal of ring artifact) for ring=0.1:
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_ring_3.sh >> output_aug_25_2023_train_ring_3.txt
14422094 14422096 14422098 14422099 14422100 14422101 14422102 14422103 14422104
EVAL:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_ring_3.sh >> output_aug_25_2023_train_ring_3_eval.txt
Analyzing results in a sweep:
export JOB_ID_ARRAY=(14422094 14422096 14422098 14422099 14422100 14422101 14422102 14422103 14422104)
export INPUT_FILE="analyze_sweep_input_4.txt"
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/analyze_sweep.sh $JOB_ID_ARRAY $INPUT_FILE
Submitted batch job 14597708

New set of foam jobs:
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_25_2023_train_foam2.txt
14432325 14432331 14432333 14432336 14432338 14432342 14432344 14432347 14432348
EVAL:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_25_2023_train_foam2_eval.txt
Analyzing results in a sweep:
export JOB_ID_ARRAY=(14432325 14432331 14432333 14432336 14432338 14432342 14432344 14432347 14432348)
export INPUT_FILE="analyze_sweep_input_5.txt"
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/analyze_sweep.sh $JOB_ID_ARRAY $INPUT_FILE
Submitted batch job 14597711

5 more trials:
cd $SCRATCH/output_CT_NVAE
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_25_2023_train_trial0.txt
14433032 14433035 14433039 14433040 14433041 14433042 14433043 14433044 14433045
EVAL:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_25_2023_train_trial0_eval.txt
Analyzing results in a sweep:
export JOB_ID_ARRAY=(14433032 14433035 14433039 14433040 14433041 14433042 14433043 14433044 14433045)
export INPUT_FILE="analyze_sweep_input_6.txt"
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/analyze_sweep.sh $JOB_ID_ARRAY $INPUT_FILE
Submitted batch job 14597744

cd $SCRATCH/output_CT_NVAE
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_25_2023_train_trial1.txt
14433055 14433058 14433060 14433061 14433062 14433063 14433064 14433065 14433066
EVAL:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_25_2023_train_trial1_eval.txt
Analyzing results in a sweep:
export JOB_ID_ARRAY=(14433055 14433058 14433060 14433061 14433062 14433063 14433064 14433065 14433066)
export INPUT_FILE="analyze_sweep_input_7.txt"
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/analyze_sweep.sh $JOB_ID_ARRAY $INPUT_FILE
Submitted batch job 14597747

cd $SCRATCH/output_CT_NVAE
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_25_2023_train_trial2.txt
14433108 14433111 14433112 14433113 14433114 14433115 14433117 14433118 14433119
EVAL:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_25_2023_train_trial2_eval.txt
Analyzing results in a sweep:
export JOB_ID_ARRAY=(14433108 14433111 14433112 14433113 14433114 14433115 14433117 14433118 14433119)
export INPUT_FILE="analyze_sweep_input_8.txt"
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/analyze_sweep.sh $JOB_ID_ARRAY $INPUT_FILE
Submitted batch job 14597764

cd $SCRATCH/output_CT_NVAE
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_25_2023_train_trial3.txt
14433149 14433152 14433153 14433155 14433157 14433160 14433161 14433163 14433164
EVAL:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_25_2023_train_trial3_eval.txt
Analyzing results in a sweep:
export JOB_ID_ARRAY=(14433149 14433152 14433153 14433155 14433157 14433160 14433161 14433163 14433164)
export INPUT_FILE="analyze_sweep_input_9.txt"
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/analyze_sweep.sh $JOB_ID_ARRAY $INPUT_FILE
Submitted batch job 14597766

cd $SCRATCH/output_CT_NVAE
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_25_2023_train_trial4.txt
14433184 14433186 14433187 14433189 14433190 14433192 14433193 14433194 14433196
EVAL:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_25_2023_train_trial4_eval.txt
Analyzing results in a sweep:
export JOB_ID_ARRAY=(14433184 14433186 14433187 14433189 14433190 14433192 14433193 14433194 14433196)
export INPUT_FILE="analyze_sweep_input_10.txt"
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/analyze_sweep.sh $JOB_ID_ARRAY $INPUT_FILE
Submitted batch job 14597767
eval test:
EVAL 2:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam.sh >> output_aug_25_2023_train_trial4_eval_test.txt



Analysis of all the last ones:
export JOB_ID_ARRAY=(14416827 14432348 14433045 14433066 14433119 14433164 14433196)
export INPUT_FILE="analyze_sweep_input_11.txt"
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/analyze_sweep.sh $JOB_ID_ARRAY $INPUT_FILE
Submitted batch job 14651389

FOAM JOBS END ##############################################################################################################
Analyzing results:

```
module load python
conda activate tomopy
salloc -N 1 --time=120 -C gpu -A m2859_g --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
cd $WORKING_DIR
python $CT_NVAE_PATH/metrics/analyze_training_results.py --expr_id 14374863 --original_size 128 --algorithm gridrec --dataset_type valid
```

Jobs analyzing the train dataset:
14639959         PD vidyagan  CT_NVAE_ANAL  1        2:00:00       0:00  2023-08-29T15:45:04  gpu_regular     N/A                  gpu&a100&hbm40 (Priority)     
14639987         PD vidyagan  CT_NVAE_ANAL  1        2:00:00       0:00  2023-08-29T15:46:33  gpu_regular     N/A                  gpu&a100&hbm40 (Priority)     
14639983         PD vidyagan  CT_NVAE_ANAL  1        2:00:00       0:00  2023-08-29T15:46:18  gpu_regular     N/A                  gpu&a100&hbm40 (Priority)     
14639980         PD vidyagan  CT_NVAE_ANAL  1        2:00:00       0:00  2023-08-29T15:46:12  gpu_regular     N/A                  gpu&a100&hbm40 (Priority)     
14639979         PD vidyagan  CT_NVAE_ANAL  1        2:00:00       0:00  2023-08-29T15:46:04  gpu_regular     N/A                  gpu&a100&hbm40 (Priority)     
14639976         PD vidyagan  CT_NVAE_ANAL  1        2:00:00       0:00  2023-08-29T15:45:57  gpu_regular     N/A                  gpu&a100&hbm40 (Priority)     
14639974         PD vidyagan  CT_NVAE_ANAL  1        2:00:00       0:00  2023-08-29T15:45:45  gpu_regular     N/A                  gpu&a100&hbm40 (Priority)     


# interactive session to test out the covid dataset
export NERSC_GPU_ALLOCATION=m2859_g

export DATA_TYPE=covid
export NUM_EXAMPLES=100
export SAVE_NAME=test_covid

export EPOCHS=1000
export RING_VAL=0
export RING=False
export BATCH_SIZE=1
export SAVE_INTERVAL=1000
export NUM_NODES=1
export USE_H5=True
export NUM_SPARSE_ANGLES=180

conda deactivate
module purge
module load python
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR

salloc -N 1 -n 1 --time=120 -C gpu -A $NERSC_GPU_ALLOCATION --qos=interactive --cpus-per-task=128

export PNM=$((10000/$NUM_SPARSE_ANGLES))
# export DATASET_ID=${DATA_TYPE}_${NUM_SPARSE_ANGLES}ang_${NUM_EXAMPLES}ex_${RING_VAL}ring
export DATASET_ID=${DATA_TYPE}_${NUM_SPARSE_ANGLES}ang_${NUM_EXAMPLES}ex

export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
export MASTER_ADDR=$(hostname)
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH

echo $MASTER_ADDR

echo "Using brain or covid data parameters"
  export NUM_LATENT_SCALES=3 # 4 or more causes a bug
  export NUM_GROUPS_PER_SCALE=10 # 16 causes out of memory error
  export NUM_POSTPROCESS_CELLS=2
  export NUM_PREPROCESS_CELLS=2
  export NUM_CELL_PER_COND_ENC=2
  export NUM_CELL_PER_COND_DEC=2
  export NUM_LATENT_PER_GROUP=20
  export NUM_PREPROCESS_BLOCKS=1
  export NUM_POSTPROCESS_BLOCKS=1
  export WEIGHT_DECAY_NORM=1e-2
  export NUM_CHANNELS_ENC=30
  export NUM_CHANNELS_DEC=30
  export NUM_NF=2
  export MIN_GROUPS_PER_SCALE=4
  export WEIGHT_DECAY_NORM_INIT=1.
  export WEIGHT_DECAY_NORM_ANNEAL=True

python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $SAVE_NAME --dataset $DATASET_ID --batch_size $BATCH_SIZE --epochs $EPOCHS --num_latent_scales $NUM_LATENT_SCALES --num_groups_per_scale $NUM_GROUPS_PER_SCALE --num_postprocess_cells $NUM_POSTPROCESS_CELLS --num_preprocess_cells $NUM_PREPROCESS_CELLS --num_cell_per_cond_enc $NUM_CELL_PER_COND_ENC --num_cell_per_cond_dec $NUM_CELL_PER_COND_DEC --num_latent_per_group $NUM_LATENT_PER_GROUP --num_preprocess_blocks $NUM_PREPROCESS_BLOCKS --num_postprocess_blocks $NUM_POSTPROCESS_BLOCKS --weight_decay_norm $WEIGHT_DECAY_NORM --num_channels_enc $NUM_CHANNELS_ENC --num_channels_dec $NUM_CHANNELS_DEC --num_nf $NUM_NF  --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax --pnm $PNM --save_interval $SAVE_INTERVAL --cont_training --model_ring_artifact $RING --num_proc_node $NUM_NODES --use_h5 $USE_H5 --min_groups_per_scale $MIN_GROUPS_PER_SCALE --weight_decay_norm_anneal $WEIGHT_DECAY_NORM_ANNEAL --weight_decay_norm_init $WEIGHT_DECAY_NORM_INIT --use_nersc

# August 29, 2023

module load python
conda activate tomopy
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
cd $WORKING_DIR

python $CT_NVAE_PATH/metrics/analyze_num_angles_sweep.py --dataset_type train
python $CT_NVAE_PATH/metrics/analyze_num_angles_sweep.py --dataset_type train --metric PSNR

# August 30, 2023

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh >> output_aug_30_2023_2.txt

JOBID            ST USER      NAME          NODES TIME_LIMIT       TIME  SUBMIT_TIME          QOS             START_TIME           FEATURES       NODELIST(REASON
14694614         PD vidyagan  CT_NVAE       1           6:00       0:00  2023-08-30T18:15:25  debug_preempt   2023-08-30T18:18:35  gpu&a100&hbm40 (Priority)     
14694624         PD vidyagan  CT_NVAE       1           6:00       0:00  2023-08-30T18:15:41  debug_preempt   2023-08-30T18:18:35  gpu&a100&hbm40 (Priority)     
14694626         PD vidyagan  CT_NVAE       1           6:00       0:00  2023-08-30T18:15:43  debug_preempt   N/A                  gpu&a100&hbm40 (Dependency)   
14694621         PD vidyagan  CT_NVAE       1           6:00       0:00  2023-08-30T18:15:35  debug_preempt   N/A                  gpu&a100&hbm40 (Dependency) 

FULL FOAM SWEEP:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh >> output_aug_30_2023_foam_2.txt
14694992 14695013 14695031 14695044 14695056 14695068 14695079 14695089 14695097
Analyzing results in a sweep:
export JOB_ID_ARRAY=(14694992 14695013 14695031 14695044 14695056 14695068 14695079 14695089 14695097)
export INPUT_FILE="aug_31_analyze_sweep_input_1.txt"
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/analyze_sweep.sh $JOB_ID_ARRAY $INPUT_FILE
Submitted batch job 14729519

14694992
14695056

Try again with 3 nodes:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh >> output_aug_30_2023_foam_3.txt
14701273 14701280 14701286 14701293 14701299 14701306 14701312 14701318 14701324
Analyzing results in a sweep:
export JOB_ID_ARRAY=(14701273 14701280 14701286 14701293 14701299 14701306 14701312 14701318 14701324)
export INPUT_FILE="aug_31_analyze_sweep_input_2.txt"
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/analyze_sweep.sh $JOB_ID_ARRAY $INPUT_FILE
Submitted batch job 14729532


# August 31, 2023
3 nodes, improved code to put eval at the end, and implemented model_candidate:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh >> output_aug_31_2023_foam_4.txt

The "OK" job is running too many times, changed script and trying for 10 epochs + final analysis:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh >> output_sept_1_2023_foam_7.txt

redo:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh >> output_sept_1_2023_foam_8.txt

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh >> output_sept_1_2023_foam_9.txt
jobs are hanging at the last epoch, add lots of print statements to the train loop to find where it is hanging
reloading from best checkpoint, not latest, so possible problems in getting full number of iters, but this could be a feature, not a bug
a list of notok means that every other job will be run if submitting multiple jobs, FIXED 9/2/2023
investigate more

# Sept 2, 2023

Trying again with fix:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh >> output_sept_2_2023_foam_0.txt

Starting one with 500 epochs:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh >> output_sept_2_2023_foam_500epoch.txt
14844050 14844070 14844090 14844107 14844118 14844129 14844139 14844146 14844153 

Debugging COVID:
Create smaller datasets:

Create the 10 example covid image dataset:

SETUP
=============================
module load python
export NERSC_GPU_ALLOCATION=m2859_g
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR
export NUM_EXAMPLES=10
=============================

export NUM_SPARSE_ANGLES=180
export RANDOM_ANGLES=True
export RING=0
export ALGORITHM=gridrec
export DO_PART_ONE=False
export DO_PART_TWO=True
export DATA_TYPE=covid
export IMAGE_ID=covid_10ex
export DATASET_ID=covid_180ang_10ex_gridrec

sbatch --time=02:00:00 -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO

Submitted batch job 14846819 (10 examples, tv)
Submitted batch job 14847734 (100 examples, gridrec)
Submitted batch job 14857580 (10 examples, gridrec)


# --use nersc vs not:

not:
09/02 11:25:33 AM (Elapsed: 00:00:02) param size = 34.312862M 
09/02 11:25:33 AM (Elapsed: 00:00:02) groups per scale: [10, 5], total_groups: 15
09/02 11:25:33 AM (Elapsed: 00:00:02) epoch 0
09/02 11:25:33 AM (Elapsed: 00:00:02) pnm_implement 10
09/02 11:27:31 AM (Elapsed: 00:02:00) train 249 1847.724731
09/02 11:29:18 AM (Elapsed: 00:03:46) train 499 292.386902
09/02 11:31:03 AM (Elapsed: 00:05:31) train 749 -243.166107
09/02 11:32:50 AM (Elapsed: 00:07:18) train 999 -514.509094
saving image: /pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-test_foam/input_image_rank_0_999.png
saving image: /pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-test_foam/ground_truth_rank_0_999.png
saving image: /pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-test_foam/sinogram_reconstruction_rank_0_999.png
saving image: /pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-test_foam/phantom_reconstruction_rank_0_999.png
09/02 11:34:36 AM (Elapsed: 00:09:04) train 1249 -678.989502

--use_nersc
09/02 11:37:01 AM (Elapsed: 00:00:01) param size = 34.312862M 
09/02 11:37:01 AM (Elapsed: 00:00:01) groups per scale: [10, 5], total_groups: 15
09/02 11:37:01 AM (Elapsed: 00:00:01) epoch 0
09/02 11:37:01 AM (Elapsed: 00:00:01) pnm_implement 10
09/02 11:38:59 AM (Elapsed: 00:01:59) train 249 1847.721802
09/02 11:40:43 AM (Elapsed: 00:03:43) train 499 292.386566
09/02 11:42:27 AM (Elapsed: 00:05:27) train 749 -243.166733
09/02 11:44:12 AM (Elapsed: 00:07:12) train 999 -514.511353

Conclusion: switched to --use_nersc


# Sept 3, 2023

Start 4 more trials of the foam sweep:

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh >> output_sept_2_2023_foam_500epoch_1.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh >> output_sept_2_2023_foam_500epoch_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh >> output_sept_2_2023_foam_500epoch_3.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh >> output_sept_2_2023_foam_500epoch_4.txt



Start covid jobs, tv and gridrec:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/train_covid_tv_gridrec.sh gridrec >> output_sept_3_2023_covid_500epoch_gridrec.txt
14891137 14891140 14891142 14891144 14891146 14891147

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/train_covid_tv_gridrec.sh tv >> output_sept_3_2023_covid_500epoch_tv.txt
14891186 14891188 14891190 14891192 14891193 14891194

Upload breadcrumb:
/Users/VGanapati/Library/CloudStorage/Dropbox/Github/CT_VAE/bread_clean.h5
/pscratch/sd/v/vidyagan/output_CT_NVAE

scp /Users/VGanapati/Library/CloudStorage/Dropbox/Github/CT_VAE/bread_clean.h5 vidyagan@saul-p1.nersc.gov:/pscratch/sd/v/vidyagan/output_CT_NVAE


Experiments for breadcrumb:
Different angles for each slice
Same angles for all slices - uniform and non-uniform

Making 10 example foam datasets
Making 100 example foam datasets


SETUP
=============================
module load python
conda activate tomopy

export NERSC_GPU_ALLOCATION=m2859_g
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR

=============================

export NUM_EXAMPLES=100
export NUM_SPARSE_ANGLES=180
export RANDOM_ANGLES=True
export RING=0
export ALGORITHM=gridrec
export DO_PART_ONE=True
export DO_PART_TWO=False
export DATA_TYPE=foam
export IMAGE_ID=foam_${NUM_EXAMPLES}ex
export DATASET_ID=foam_${NUM_SPARSE_ANGLES}ang_${NUM_EXAMPLES}ex

sbatch --time=00:30:00 -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO

Submitted batch job 14893278 (10 examples) DONE
Submitted batch job 14893295 (100 examples) DONE


Create dataset sweeps:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 10 >> output_sept_3_2023_foam_10dataset.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 100 >> output_sept_3_2023_foam_100dataset.txt

# Sept 5, 2023


module load python
conda activate tomopy
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
cd $WORKING_DIR

python $CT_NVAE_PATH/metrics/analyze_num_angles_sweep.py --dataset_type train
python $CT_NVAE_PATH/metrics/analyze_num_angles_sweep.py --dataset_type train --metric PSNR
python $CT_NVAE_PATH/metrics/analyze_num_angles_sweep.py --dataset_type train --metric SSIM

COVID results:
looks like the tv initial conditions were better

Sweep the 10 example foam datasets:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 >> output_sept_5_2023_foam2_10ex_0.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 >> output_sept_5_2023_foam2_10ex_1.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 >> output_sept_5_2023_foam2_10ex_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 >> output_sept_5_2023_foam2_10ex_3.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 >> output_sept_5_2023_foam2_10ex_4.txt
15025584 15025624 15025637 15025647 15025658 15025673 15025688 15025703 15025718 

Sweep the 100 example foam datasets:

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 >> output_sept_5_2023_foam3_100ex_0.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 >> output_sept_5_2023_foam3_100ex_1.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 >> output_sept_5_2023_foam2_100ex_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 >> output_sept_5_2023_foam2_100ex_3.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 >> output_sept_5_2023_foam2_100ex_4.txt
15025651 15025666 15025680 15025692 15025707 15025721 15025745 15025759 15025800 

# Sept 7, 2023

module load python
conda activate tomopy
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
cd $WORKING_DIR

python $CT_NVAE_PATH/metrics/analyze_num_angles_sweep.py --dataset_type train
python $CT_NVAE_PATH/metrics/analyze_num_angles_sweep.py --dataset_type train --metric PSNR
python $CT_NVAE_PATH/metrics/analyze_num_angles_sweep.py --dataset_type train --metric SSIM


Do the sweeps again, except multiply the number of epochs by 10 and 100:

Sweep the 10 example foam datasets:
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 >> output_sept_7_2023_foam_10ex_0.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 >> output_sept_7_2023_foam_10ex_1.txt (output file in overall folder)
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 >> output_sept_7_2023_foam_10ex_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 >> output_sept_7_2023_foam_10ex_3.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 >> output_sept_7_2023_foam_10ex_4.txt

Sweep the 100 example foam datasets:

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 >> output_sept_7_2023_foam_100ex_0.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 >> output_sept_7_2023_foam_100ex_1.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 >> output_sept_7_2023_foam_100ex_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 >> output_sept_7_2023_foam_100ex_3.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 >> output_sept_7_2023_foam_100ex_4.txt


Creation of foam datasets, uniform, tv:

cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 100 False False 0 >> output_sept_8_2023_foam_uniform.txt

Creation of foam datasets, random, tv:

cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 100 True False 0 >> output_sept_8_2023_foam_random.txt

Creation of foam datasets, random constant, tv:

cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 100 True True 0 >> output_sept_8_2023_foam_random_constant0.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 100 True True 1 >> output_sept_8_2023_foam_random_constant1.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 100 True True 2 >> output_sept_8_2023_foam_random_constant2.txt

# Sept 8, 2023

Sweep all 5 datasets 5 times: DONE
cd $WORKING_DIR
Uniform
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_8_2023_foam_100ex_train_0.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_8_2023_foam_100ex_train_1.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_8_2023_foam_100ex_train_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_8_2023_foam_100ex_train_3.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_8_2023_foam_100ex_train_4.txt

Random, Changing
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_8_2023_foam_100ex_train_5.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_8_2023_foam_100ex_train_6.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_8_2023_foam_100ex_train_7.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_8_2023_foam_100ex_train_8.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_8_2023_foam_100ex_train_9.txt

Random, Same, Try 1
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_8_2023_foam_100ex_train_10.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_8_2023_foam_100ex_train_11.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_8_2023_foam_100ex_train_12.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_8_2023_foam_100ex_train_13.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_8_2023_foam_100ex_train_14.txt

Random, Same, Try 2
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_8_2023_foam_100ex_train_15.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_8_2023_foam_100ex_train_16.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_8_2023_foam_100ex_train_17.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_8_2023_foam_100ex_train_18.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_8_2023_foam_100ex_train_19.txt

Random, Same, Try 3
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_8_2023_foam_100ex_train_20.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_8_2023_foam_100ex_train_21.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_8_2023_foam_100ex_train_22.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_8_2023_foam_100ex_train_23.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_8_2023_foam_100ex_train_24.txt


SETUP:
================================
module load python
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
export NERSC_GPU_ALLOCATION=m2859_g
================================
salloc -N 1 -n 1 --time=120 -C gpu -A $NERSC_GPU_ALLOCATION --qos=interactive --cpus-per-task=128

Analyzing the breadcrumb dataset
python $CT_NVAE_PATH/computed_tomography/tests/test_bread_crumb.py

Analyzing the covid dataset

python $CT_NVAE_PATH/computed_tomography/tests/test_large_tomopy_reconstruction.py

# September 11, 2023

Sweep with 1 normalizing flow: Hardcoded in train_multi_node_preempt, REMEMBER TO CHANGE BACK (changed back on Oct 18, 2023)
Random, Changing
cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_11_2023_foam_100ex_train_0.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_11_2023_foam_100ex_train_1.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_11_2023_foam_100ex_train_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_11_2023_foam_100ex_train_3.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_11_2023_foam_100ex_train_4.txt

# September 13, 2023

REDO all sweeps with normalizing flow:

export NERSC_GPU_ALLOCATION=m3562_g
cd $WORKING_DIR


Uniform
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_13_2023_foam_100ex_train_0.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_13_2023_foam_100ex_train_1.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_13_2023_foam_100ex_train_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_13_2023_foam_100ex_train_3.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_13_2023_foam_100ex_train_4.txt

Random, Changing
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_13_2023_foam_100ex_train_5.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_13_2023_foam_100ex_train_6.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_13_2023_foam_100ex_train_7.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_13_2023_foam_100ex_train_8.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_13_2023_foam_100ex_train_9.txt

Random, Same, Try 1
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_13_2023_foam_100ex_train_10.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_13_2023_foam_100ex_train_11.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_13_2023_foam_100ex_train_12.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_13_2023_foam_100ex_train_13.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_13_2023_foam_100ex_train_14.txt

Random, Same, Try 2
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_13_2023_foam_100ex_train_15.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_13_2023_foam_100ex_train_16.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_13_2023_foam_100ex_train_17.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_13_2023_foam_100ex_train_18.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_13_2023_foam_100ex_train_19.txt

Random, Same, Try 3
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_13_2023_foam_100ex_train_20.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_13_2023_foam_100ex_train_21.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_13_2023_foam_100ex_train_22.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_13_2023_foam_100ex_train_23.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_13_2023_foam_100ex_train_24.txt

Complete another round for more examples:

Uniform
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_13_2023_2_foam_100ex_train_0.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_13_2023_2_foam_100ex_train_1.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_13_2023_2_foam_100ex_train_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_13_2023_2_foam_100ex_train_3.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_13_2023_2_foam_100ex_train_4.txt

Random, Changing
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_13_2023_2_foam_100ex_train_5.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_13_2023_2_foam_100ex_train_6.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_13_2023_2_foam_100ex_train_7.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_13_2023_2_foam_100ex_train_8.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_13_2023_2_foam_100ex_train_9.txt

Random, Same, Try 1
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_13_2023_2_foam_100ex_train_10.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_13_2023_2_foam_100ex_train_11.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_13_2023_2_foam_100ex_train_12.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_13_2023_2_foam_100ex_train_13.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_13_2023_2_foam_100ex_train_14.txt

Random, Same, Try 2
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_13_2023_2_foam_100ex_train_15.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_13_2023_2_foam_100ex_train_16.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_13_2023_2_foam_100ex_train_17.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_13_2023_2_foam_100ex_train_18.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_13_2023_2_foam_100ex_train_19.txt

Random, Same, Try 3
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_13_2023_2_foam_100ex_train_20.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_13_2023_2_foam_100ex_train_21.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_13_2023_2_foam_100ex_train_22.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_13_2023_2_foam_100ex_train_23.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_13_2023_2_foam_100ex_train_24.txt



Make the following Covid datasets:

10 examples, TV with 10 iterations
Sparse Angles to use: 20, 40, 60, 80, 100, 120, 140, 160, 180
Poisson Noise Level to use (numerator): 10000 (no change)


Creation of Covid datasets, uniform, tv:

cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 100 False False 0 >> output_sept_13_2023_covid_uniform.txt

Creation of covid datasets, random, tv:

cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 100 True False 0 >> output_sept_13_2023_covid_random.txt

Creation of covid datasets, random constant, tv:

cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 100 True True 0 >> output_sept_13_2023_covid_random_constant0.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 100 True True 1 >> output_sept_13_2023_covid_random_constant1.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 100 True True 2 >> output_sept_13_2023_covid_random_constant2.txt

REDO for 10 examples each:
Added PNM_NUM to arguments

cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 10 False False 0 10000 >> output_sept_14_2023_covid_uniform2.txt

Creation of covid datasets, random, tv:

cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 10 True False 0 10000 >> output_sept_14_2023_covid_random2.txt

Creation of covid datasets, random constant, tv:

cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 10 True True 0 10000 >> output_sept_14_2023_covid_random_constant02.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 10 True True 1 10000 >> output_sept_14_2023_covid_random_constant12.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 10 True True 2 10000 >> output_sept_14_2023_covid_random_constant22.txt


# September 14, 2023
Increased SNR, pnm numerator to 1000000


Creation of covid datasets, random, tv:

cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 10 False False 0_pnm_6 1000000 >> output_sept_14_2023_covid_uniform2.txt

Creation of covid datasets, random, tv:

cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 10 True False 0_pnm_6 1000000 >> output_sept_14_2023_covid_random2.txt

Creation of covid datasets, random constant, tv:

cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 10 True True 0_pnm_6 1000000 >> output_sept_14_2023_covid_random_constant02.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 10 True True 1_pnm_6 1000000 >> output_sept_14_2023_covid_random_constant12.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets_foam.sh 10 True True 2_pnm_6 1000000 >> output_sept_14_2023_covid_random_constant22.txt



Start the covid sweeps, 100 examples: RUNNING

cd $WORKING_DIR
Uniform
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_14_2023_covid_100ex_train_0.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_14_2023_covid_100ex_train_1.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_14_2023_covid_100ex_train_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_14_2023_covid_100ex_train_3.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 False False 0 >> output_sept_14_2023_covid_100ex_train_4.txt

Random, Changing
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_14_2023_covid_100ex_train_5.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_14_2023_covid_100ex_train_6.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_14_2023_covid_100ex_train_7.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_14_2023_covid_100ex_train_8.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True False 0 >> output_sept_14_2023_covid_100ex_train_9.txt

Random, Same, Try 1
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_14_2023_covid_100ex_train_10.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_14_2023_covid_100ex_train_11.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_14_2023_covid_100ex_train_12.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_14_2023_covid_100ex_train_13.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 0 >> output_sept_14_2023_covid_100ex_train_14.txt

Random, Same, Try 2
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_13_2023_2_covid_100ex_train_15.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_13_2023_2_covid_100ex_train_16.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_13_2023_2_covid_100ex_train_17.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_13_2023_2_covid_100ex_train_18.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 1 >> output_sept_13_2023_2_covid_100ex_train_19.txt

Random, Same, Try 3
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_13_2023_2_covid_100ex_train_20.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_13_2023_2_covid_100ex_train_21.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_13_2023_2_covid_100ex_train_22.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_13_2023_2_covid_100ex_train_23.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 100 True True 2 >> output_sept_13_2023_2_covid_100ex_train_24.txt

# September 15, 2023

10 examples, regular noise:

cd $WORKING_DIR
Uniform
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 False False 0 10000 >> output_sept_15_2023_covid_10ex_train_0.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 False False 0 10000 >> output_sept_15_2023_covid_10ex_train_1.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 False False 0 10000 >> output_sept_15_2023_covid_10ex_train_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 False False 0 10000 >> output_sept_15_2023_covid_10ex_train_3.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 False False 0 10000 >> output_sept_15_2023_covid_10ex_train_4.txt (RUN ME)

Random, Changing
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True False 0 10000 >> output_sept_15_2023_covid_10ex_train_5.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True False 0 10000 >> output_sept_15_2023_covid_10ex_train_6.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True False 0 10000 >> output_sept_15_2023_covid_10ex_train_7.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True False 0 10000 >> output_sept_15_2023_covid_10ex_train_8.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True False 0 10000 >> output_sept_15_2023_covid_10ex_train_9.txt (RUN ME)

Random, Same, Try 1
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 0 10000 >> output_sept_15_2023_covid_10ex_train_10.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 0 10000 >> output_sept_15_2023_covid_10ex_train_11.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 0 10000 >> output_sept_15_2023_covid_10ex_train_12.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 0 10000 >> output_sept_15_2023_covid_10ex_train_13.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 0 10000 >> output_sept_15_2023_covid_10ex_train_14.txt (RUN ME)

Random, Same, Try 2
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 1 10000 >> output_sept_15_2023_2_covid_10ex_train_15.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 1 10000 >> output_sept_15_2023_2_covid_10ex_train_16.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 1 10000 >> output_sept_15_2023_2_covid_10ex_train_17.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 1 10000 >> output_sept_15_2023_2_covid_10ex_train_18.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 1 10000 >> output_sept_15_2023_2_covid_10ex_train_19.txt (RUN ME)

Random, Same, Try 3
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 2 10000 >> output_sept_15_2023_2_covid_10ex_train_20.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 2 10000 >> output_sept_15_2023_2_covid_10ex_train_21.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 2 10000 >> output_sept_15_2023_2_covid_10ex_train_22.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 2 10000 >> output_sept_15_2023_2_covid_10ex_train_23.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 2 10000 >> output_sept_15_2023_2_covid_10ex_train_24.txt (RUN ME)

10 examples, improved SNR:

cd $WORKING_DIR
Uniform
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 False False 0_pnm_6 1000000 >> output_sept_15_2023_covid_10ex_train_0_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 False False 0_pnm_6 1000000 >> output_sept_15_2023_covid_10ex_train_1_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 False False 0_pnm_6 1000000 >> output_sept_15_2023_covid_10ex_train_2_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 False False 0_pnm_6 1000000 >> output_sept_15_2023_covid_10ex_train_3_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 False False 0_pnm_6 1000000 >> output_sept_15_2023_covid_10ex_train_4_2.txt (RUN ME)

Random, Changing
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True False 0_pnm_6 1000000 >> output_sept_15_2023_covid_10ex_train_5_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True False 0_pnm_6 1000000 >> output_sept_15_2023_covid_10ex_train_6_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True False 0_pnm_6 1000000 >> output_sept_15_2023_covid_10ex_train_7_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True False 0_pnm_6 1000000 >> output_sept_15_2023_covid_10ex_train_8_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True False 0_pnm_6 1000000 >> output_sept_15_2023_covid_10ex_train_9_2.txt (RUN ME)

Random, Same, Try 1
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 0_pnm_6 1000000 >> output_sept_15_2023_covid_10ex_train_10_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 0_pnm_6 1000000 >> output_sept_15_2023_covid_10ex_train_11_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 0_pnm_6 1000000 >> output_sept_15_2023_covid_10ex_train_12_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 0_pnm_6 1000000 >> output_sept_15_2023_covid_10ex_train_13_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 0_pnm_6 1000000 >> output_sept_15_2023_covid_10ex_train_14_2.txt (RUN ME)

Random, Same, Try 2
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 1_pnm_6 1000000 >> output_sept_15_2023_2_covid_10ex_train_15_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 1_pnm_6 1000000 >> output_sept_15_2023_2_covid_10ex_train_16_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 1_pnm_6 1000000 >> output_sept_15_2023_2_covid_10ex_train_17_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 1_pnm_6 1000000 >> output_sept_15_2023_2_covid_10ex_train_18_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 1_pnm_6 1000000 >> output_sept_15_2023_2_covid_10ex_train_19_2.txt (RUN ME)

Random, Same, Try 3
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 2_pnm_6 1000000 >> output_sept_15_2023_2_covid_10ex_train_20_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 2_pnm_6 1000000 >> output_sept_15_2023_2_covid_10ex_train_21_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 2_pnm_6 1000000 >> output_sept_15_2023_2_covid_10ex_train_22_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 2_pnm_6 1000000 >> output_sept_15_2023_2_covid_10ex_train_23_2.txt
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_foam_slurm_dep.sh 10 True True 2_pnm_6 1000000 >> output_sept_15_2023_2_covid_10ex_train_24_2.txt (RUN ME)

# October 13, 2023

module load python
conda activate tomopy
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
cd $WORKING_DIR
export NERSC_GPU_ALLOCATION=m2859_g

python $CT_NVAE_PATH/metrics/analyze_num_angles_sweep.py --dataset_type train

Covid reconstructions that are reasonable:
/pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824058
/pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824079
/pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824101
/pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824115
/pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824163
/pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824177
/pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824184
/pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824191
Really good: /pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824261
Really good: /pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824302
Really good: /pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824309
Really good: /pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824331
Really good: /pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824338
Really good: /pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824346
Really good: /pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824354
Really good: /pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts/eval-15824398

# October 18, 2023

## Full pipeline:


### Creation of the base images

module load python
conda activate tomopy

export NERSC_GPU_ALLOCATION=m2859_g
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR

export NUM_EXAMPLES=100 # 100, 10
export NUM_SPARSE_ANGLES=180
export RANDOM_ANGLES=True
export RING=0
export ALGORITHM=tv
export DO_PART_ONE=True
export DO_PART_TWO=False
export DATA_TYPE=foam # foam, covid
export IMAGE_ID=${DATA_TYPE}_${NUM_EXAMPLES}ex
export DATASET_ID=${DATA_TYPE}_${NUM_SPARSE_ANGLES}ang_${NUM_EXAMPLES}ex_${ALGORITHM}
export CONSTANT_ANGLES=False
export PNM_NUM=10000 # 10000, 1000000

sbatch --time=02:00:00 -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $CONSTANT_ANGLES $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO $PNM_NUM

### Sweep of dataset creation, varying number of angles, tv algorithm hardcoded in

#### Uniform angles
cd $WORKING_DIR

export NUM_EXAMPLES=100 # 100, 10
export RANDOM_ANGLES=False
export CONSTANT_ANGLES=False
export TAG=0 # 0, 0_pnm_6 # increment 0 for each trial
export PNM_NUM=10000 # 10000, 1000000
export DATA_TYPE=foam # foam, covid

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE >> output_sept_8_2023_foam_uniform.txt



#### Random angles
cd $WORKING_DIR

export NUM_EXAMPLES=100 # 100, 10
export RANDOM_ANGLES=True
export CONSTANT_ANGLES=False
export TAG=0 # 0, 0_pnm_6 # increment 0 for each trial
export PNM_NUM=10000 # 10000, 1000000
export DATA_TYPE=foam # foam, covid

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE >> output_sept_8_2023_foam_random.txt

#### Random constant (can do multiple trials of dataset creation,change the $TAG and the output file name)

cd $WORKING_DIR

export NUM_EXAMPLES=100 # 100, 10
export RANDOM_ANGLES=True
export CONSTANT_ANGLES=True
export TAG=0 # 0, 0_pnm_6 # increment 0 for each trial
export PNM_NUM=10000 # 10000, 1000000
export DATA_TYPE=foam # foam, covid

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE >> output_sept_8_2023_foam_random_constant0.txt

### Run the training (can be run in parallel for multiple trials, just change output file name)

#### Uniform
cd $WORKING_DIR

export NUM_EXAMPLES=100 # 100, 10
export RANDOM_ANGLES=False
export CONSTANT_ANGLES=False
export TAG=0 # 0, 0_pnm_6 # increment 0 for each trial
export PNM_NUM=10000 # 10000, 1000000
export DATA_TYPE=foam # foam, covid
export BATCH_SIZE=16 # 16, 1
export NUM_NODES=3 # 3, 16
export ORIGINAL_SIZE=128 # 128, 512
export EPOCH_MULT=1000 # 1000, 500

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_slurm_dep.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE $BATCH_SIZE $NUM_NODES $ORIGINAL_SIZE $EPOCH_MULT >> output_sept_8_2023_foam_100ex_train_0.txt

#### Random
cd $WORKING_DIR

export NUM_EXAMPLES=100 # 100, 10
export RANDOM_ANGLES=True
export CONSTANT_ANGLES=False
export TAG=0 # 0, 0_pnm_6 # increment 0 for each trial
export PNM_NUM=10000 # 10000, 1000000
export DATA_TYPE=foam # foam, covid
export BATCH_SIZE=16 # 16, 1
export NUM_NODES=3 # 3, 16
export ORIGINAL_SIZE=128 # 128, 512
export EPOCH_MULT=1000 # 1000, 500

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_slurm_dep.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE $BATCH_SIZE $NUM_NODES $ORIGINAL_SIZE $EPOCH_MULT >> output_sept_8_2023_foam_100ex_train_5.txt

#### Random constant (change the $TAG as needed)

export NUM_EXAMPLES=100 # 100, 10
export RANDOM_ANGLES=True
export CONSTANT_ANGLES=True
export TAG=0 # 0, 0_pnm_6 # increment 0 for each trial
export PNM_NUM=10000 # 10000, 1000000
export DATA_TYPE=foam # foam, covid
export BATCH_SIZE=16 # 16, 1
export NUM_NODES=3 # 3, 16
export ORIGINAL_SIZE=128 # 128, 512
export EPOCH_MULT=1000 # 1000, 500

cd $WORKING_DIR
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_slurm_dep.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE $BATCH_SIZE $NUM_NODES $ORIGINAL_SIZE $EPOCH_MULT >> output_sept_8_2023_foam_100ex_train_10.txt

### Analyze the results

module load python
conda activate tomopy
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR

python $CT_NVAE_PATH/metrics/analyze_num_angles_sweep.py --dataset_type train
python $CT_NVAE_PATH/metrics/analyze_num_angles_sweep.py --dataset_type train --metric PSNR
python $CT_NVAE_PATH/metrics/analyze_num_angles_sweep.py --dataset_type train --metric SSIM


# October 20, 2023

## Add angles option (as in first CT_VAE paper):

test in `/pscratch/sd/v/vidyagan/CT_NVAE/computed_tomography/tests/test_sparse_reconstruction_mask.py`

How to run test:

module load python
conda activate tomopy
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR
python $CT_NVAE_PATH/computed_tomography/tests/test_sparse_reconstruction_mask.py

Saves reconstruction_mask.png in the current directory.

# October 23, 2023

### PIPELINE START
Test dataset creation directly (use base images from $WORKING_DIR/images_foam_10ex):
### Dataset Creation
export NUM_EXAMPLES=10 # 100, 10
export RANDOM_ANGLES=True
export CONSTANT_ANGLES=False
export TAG=test_mask # 0, 0_pnm_6 # increment 0 for each trial
export PNM_NUM=10000 # 10000, 1000000
export DATA_TYPE=foam # foam, covid

conda deactivate
module purge
module load python
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR

export RING=0
export ALGORITHM=tv
export DO_PART_ONE=False # False assumes we already have the base images
export DO_PART_TWO=True
export IMAGE_ID=${DATA_TYPE}_${NUM_EXAMPLES}ex
export DO_PART_ONE=False
export DO_PART_TWO=True
export NUM_SPARSE_ANGLES=20

export DATASET_ID=${DATA_TYPE}_${NUM_SPARSE_ANGLES}ang_${NUM_EXAMPLES}ex_${RING}ring_${ALGORITHM}_${RANDOM_ANGLES}random_${CONSTANT_ANGLES}constant${TAG}

. $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $CONSTANT_ANGLES $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO $PNM_NUM

Test training directly, new session but following from above code:

### Training

conda deactivate
module purge
module load python
export NERSC_GPU_ALLOCATION=m2859_g
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR

export NUM_EXAMPLES=10 # 100, 10
export RANDOM_ANGLES=True
export CONSTANT_ANGLES=False
export TAG=test_mask # 0, 0_pnm_6 # increment 0 for each trial
export PNM_NUM=10000 # 10000, 1000000
export DATA_TYPE=foam # foam, covid
export BATCH_SIZE=16 # 16, 1
export NUM_NODES=1 # 3, 16
export ORIGINAL_SIZE=128 # 128, 512

export RING_VAL=0
export RING=False
export SAVE_INTERVAL=5
export USE_H5=True
export NUM_SPARSE_ANGLES=20
export EPOCHS=2
export ALGORITHM=tv
export PNM=$(($PNM_NUM/$NUM_SPARSE_ANGLES))

export DATASET_ID=${DATA_TYPE}_${NUM_SPARSE_ANGLES}ang_${NUM_EXAMPLES}ex_${RING_VAL}ring_${ALGORITHM}_${RANDOM_ANGLES}random_${CONSTANT_ANGLES}constant${TAG}
export SAVE_NAME=test_mask2

export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH

export NUM_LATENT_SCALES=2
export NUM_GROUPS_PER_SCALE=10
export NUM_POSTPROCESS_CELLS=3
export NUM_PREPROCESS_CELLS=3
export NUM_CELL_PER_COND_ENC=2
export NUM_CELL_PER_COND_DEC=2
export NUM_LATENT_PER_GROUP=20
export NUM_PREPROCESS_BLOCKS=2
export NUM_POSTPROCESS_BLOCKS=2
export WEIGHT_DECAY_NORM=1e-2
export NUM_CHANNELS_ENC=32
export NUM_CHANNELS_DEC=32
export NUM_NF=0
export MIN_GROUPS_PER_SCALE=1
export WEIGHT_DECAY_NORM_INIT=10.
export WEIGHT_DECAY_NORM_ANNEAL=False

export FINAL_TRAIN=True # False
export FINAL_TEST=True # False

export NERSC_GPU_ALLOCATION=m2859_g
salloc -N $NUM_NODES -n $NUM_NODES --time=120 -C gpu -A $NERSC_GPU_ALLOCATION --qos=interactive --cpus-per-task=128

export MASTER_ADDR=$(hostname)
export NUM_PROCESS_PER_NODE=4 # 1

python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $SAVE_NAME --dataset $DATASET_ID --batch_size $BATCH_SIZE --epochs $EPOCHS --num_latent_scales $NUM_LATENT_SCALES --num_groups_per_scale $NUM_GROUPS_PER_SCALE --num_postprocess_cells $NUM_POSTPROCESS_CELLS --num_preprocess_cells $NUM_PREPROCESS_CELLS --num_cell_per_cond_enc $NUM_CELL_PER_COND_ENC --num_cell_per_cond_dec $NUM_CELL_PER_COND_DEC --num_latent_per_group $NUM_LATENT_PER_GROUP --num_preprocess_blocks $NUM_PREPROCESS_BLOCKS --num_postprocess_blocks $NUM_POSTPROCESS_BLOCKS --weight_decay_norm $WEIGHT_DECAY_NORM --num_channels_enc $NUM_CHANNELS_ENC --num_channels_dec $NUM_CHANNELS_DEC --num_nf $NUM_NF  --ada_groups --num_process_per_node $NUM_PROCESS_PER_NODE --use_se --res_dist --fast_adamax --pnm $PNM --save_interval $SAVE_INTERVAL --model_ring_artifact $RING --num_proc_node $NUM_NODES --use_h5 $USE_H5 --min_groups_per_scale $MIN_GROUPS_PER_SCALE --weight_decay_norm_anneal $WEIGHT_DECAY_NORM_ANNEAL --weight_decay_norm_init $WEIGHT_DECAY_NORM_INIT --final_train $FINAL_TRAIN --final_test $FINAL_TEST --use_nersc --use_masks True # --cont_training

### Analysis

Open a new terminal (doesn't have to be interactive node)

conda deactivate
module purge
module load python
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR

export DATASET_TYPE=train # test
export SAVE_NAME=test_mask2
export ORIGINAL_SIZE=128
export ALGORITHM=tv

python $CT_NVAE_PATH/metrics/analyze_training_results.py --expr_id $SAVE_NAME --original_size $ORIGINAL_SIZE --algorithm $ALGORITHM --dataset_type $DATASET_TYPE

### PIPELINE END

### Fix for use_masks (make 2 channels an argument instead of hard-coded)
XXX in utils.py and model.py
remove all the print statements
FIXED


# November 1, 2023

TODO: redo full batch pipeline with masks
TODO: add --use_masks True and --use_masks False to the pipeline
Compare with and without masks

# November 3, 2023

Running full pipeline (adding in masks option):

## Full pipeline:


### Creation of the base images

module load python
conda activate tomopy

export NERSC_GPU_ALLOCATION=m2859_g
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR

export NUM_EXAMPLES=100 # 100, 10
export NUM_SPARSE_ANGLES=180
export RANDOM_ANGLES=True
export RING=0
export ALGORITHM=tv
export DO_PART_ONE=True
export DO_PART_TWO=False
export DATA_TYPE=foam # foam, covid
export IMAGE_ID=${DATA_TYPE}_${NUM_EXAMPLES}ex
export DATASET_ID=${DATA_TYPE}_${NUM_SPARSE_ANGLES}ang_${NUM_EXAMPLES}ex_${ALGORITHM}
export CONSTANT_ANGLES=False
export PNM_NUM=10000 # 10000, 1000000

sbatch --time=02:00:00 -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $CONSTANT_ANGLES $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO $PNM_NUM

### Sweep of dataset creation, varying number of angles, tv algorithm hardcoded in

#### Uniform angles
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR

export NUM_EXAMPLES=100 # 100, 10
export RANDOM_ANGLES=False
export CONSTANT_ANGLES=False
export TAG=0_masks # 0, 0_pnm_6 # increment 0 for each trial
export PNM_NUM=10000 # 10000, 1000000
export DATA_TYPE=foam # foam, covid

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE >> output_nov_3_2023_foam_uniform.txt



#### Random angles
cd $WORKING_DIR

export NUM_EXAMPLES=100 # 100, 10
export RANDOM_ANGLES=True
export CONSTANT_ANGLES=False
export TAG=0_masks # 0, 0_pnm_6 # increment 0 for each trial
export PNM_NUM=10000 # 10000, 1000000
export DATA_TYPE=foam # foam, covid

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE >> output_nov_3_2023_foam_random.txt

#### Random constant (can do multiple trials of dataset creation,change the $TAG and the output file name)

cd $WORKING_DIR

export NUM_EXAMPLES=100 # 100, 10
export RANDOM_ANGLES=True
export CONSTANT_ANGLES=True
export TAG=0_masks # 0, 0_pnm_6 # increment 0 for each trial
export PNM_NUM=10000 # 10000, 1000000
export DATA_TYPE=foam # foam, covid
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE >> output_nov_3_2023_foam_random_constant0.txt

export TAG=1_masks # 0, 0_pnm_6 # increment 0 for each trial
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE >> output_nov_3_2023_foam_random_constant1.txt

export TAG=2_masks # 0, 0_pnm_6 # increment 0 for each trial
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_datasets.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE >> output_nov_3_2023_foam_random_constant2.txt



### Run the training (can be run in parallel for multiple trials, just change output file name)

#### Uniform
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR

export NUM_EXAMPLES=100 # 100, 10
export RANDOM_ANGLES=False
export CONSTANT_ANGLES=False
export TAG=0_masks # 0_masks, 0_pnm_6 # increment 0 for each trial
export PNM_NUM=10000 # 10000, 1000000
export DATA_TYPE=foam # foam, covid
export BATCH_SIZE=16 # 16, 1
export NUM_NODES=3 # 3, 16
export ORIGINAL_SIZE=128 # 128, 512
export EPOCH_MULT=1000 # 1000, 500
export USE_MASKS=True # try both True and False
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_slurm_dep.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE $BATCH_SIZE $NUM_NODES $ORIGINAL_SIZE $EPOCH_MULT $USE_MASKS >> output_nov_2023_foam_100ex_train_0.txt


export USE_MASKS=False # try both True and False
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_slurm_dep.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE $BATCH_SIZE $NUM_NODES $ORIGINAL_SIZE $EPOCH_MULT $USE_MASKS >> output_nov_2023_foam_100ex_train_1.txt

#### Random
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR

export NUM_EXAMPLES=100 # 100, 10
export RANDOM_ANGLES=True
export CONSTANT_ANGLES=False
export TAG=0_masks # 0_masks, 0_pnm_6 # increment 0 for each trial
export PNM_NUM=10000 # 10000, 1000000
export DATA_TYPE=foam # foam, covid
export BATCH_SIZE=16 # 16, 1
export NUM_NODES=3 # 3, 16
export ORIGINAL_SIZE=128 # 128, 512
export EPOCH_MULT=1000 # 1000, 500
export USE_MASKS=True # try both True and False

. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_slurm_dep.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE $BATCH_SIZE $NUM_NODES $ORIGINAL_SIZE $EPOCH_MULT $USE_MASKS >> output_sept_8_2023_foam_100ex_train_2.txt

export USE_MASKS=False # try both True and False
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_slurm_dep.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE $BATCH_SIZE $NUM_NODES $ORIGINAL_SIZE $EPOCH_MULT $USE_MASKS >> output_nov_2023_foam_100ex_train_3.txt

#### Random constant (change the $TAG as needed)
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR

export NUM_EXAMPLES=100 # 100, 10
export RANDOM_ANGLES=True
export CONSTANT_ANGLES=True
export TAG=0_masks # 0, 0_pnm_6 # increment 0 for each trial
export PNM_NUM=10000 # 10000, 1000000
export DATA_TYPE=foam # foam, covid
export BATCH_SIZE=16 # 16, 1
export NUM_NODES=3 # 3, 16
export ORIGINAL_SIZE=128 # 128, 512
export EPOCH_MULT=1000 # 1000, 500
export USE_MASKS=True # try both True and False
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_slurm_dep.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE $BATCH_SIZE $NUM_NODES $ORIGINAL_SIZE $EPOCH_MULT $USE_MASKS >> output_sept_8_2023_foam_100ex_train_4.txt

export USE_MASKS=False # try both True and False
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_slurm_dep.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE $BATCH_SIZE $NUM_NODES $ORIGINAL_SIZE $EPOCH_MULT $USE_MASKS >> output_nov_2023_foam_100ex_train_5.txt


export TAG=1_masks
export USE_MASKS=True
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_slurm_dep.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE $BATCH_SIZE $NUM_NODES $ORIGINAL_SIZE $EPOCH_MULT $USE_MASKS >> output_sept_8_2023_foam_100ex_train_6.txt

export USE_MASKS=False # try both True and False
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_slurm_dep.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE $BATCH_SIZE $NUM_NODES $ORIGINAL_SIZE $EPOCH_MULT $USE_MASKS >> output_nov_2023_foam_100ex_train_7.txt


export TAG=2_masks
export USE_MASKS=True
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_slurm_dep.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE $BATCH_SIZE $NUM_NODES $ORIGINAL_SIZE $EPOCH_MULT $USE_MASKS >> output_sept_8_2023_foam_100ex_train_8.txt

export USE_MASKS=False # try both True and False
. /pscratch/sd/v/vidyagan/CT_NVAE/slurm/sweep_num_proj_train_slurm_dep.sh $NUM_EXAMPLES $RANDOM_ANGLES $CONSTANT_ANGLES $TAG $PNM_NUM $DATA_TYPE $BATCH_SIZE $NUM_NODES $ORIGINAL_SIZE $EPOCH_MULT $USE_MASKS >> output_nov_2023_foam_100ex_train_9.txt



### Analyze the results

module load python
conda activate tomopy
export PYTHONPATH=$SCRATCH/CT_NVAE:$PYTHONPATH
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR

python $CT_NVAE_PATH/metrics/analyze_num_angles_sweep.py --dataset_type train
python $CT_NVAE_PATH/metrics/analyze_num_angles_sweep.py --dataset_type train --metric PSNR
python $CT_NVAE_PATH/metrics/analyze_num_angles_sweep.py --dataset_type train --metric SSIM


# December 1, 2023

Toy example test

Example 1:
0.1 0.2
0.3 0.4


Example 2:
0.3 0.4
0.1 0.2

Angles: 0 and pi/2

Full pipeline with the toy example:

### Creation of the dataset

module load python
conda activate tomopy

export NERSC_GPU_ALLOCATION=m2859_g
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR

salloc -N 1 -n 1 --time=120 -C gpu -A $NERSC_GPU_ALLOCATION --qos=interactive --cpus-per-task=128

export NUM_EXAMPLES=100 # 100, 10
export NUM_SPARSE_ANGLES=1
export RANDOM_ANGLES=True
export RING=0
export ALGORITHM=tv
export DO_PART_ONE=True
export DO_PART_TWO=True
export DATA_TYPE=toy # foam, covid, toy
export IMAGE_ID=${DATA_TYPE}_${NUM_EXAMPLES}ex
export DATASET_ID=${DATA_TYPE}_${NUM_SPARSE_ANGLES}ang_${NUM_EXAMPLES}ex_${RING}ring_${ALGORITHM}_${RANDOM_ANGLES}random_${CONSTANT_ANGLES}constant${TAG}
export CONSTANT_ANGLES=False
export PNM_NUM=10000 # 10000, 1000000

. $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $CONSTANT_ANGLES $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO $PNM_NUM

export RANDOM_ANGLES=False
export DATASET_ID=${DATA_TYPE}_${NUM_SPARSE_ANGLES}ang_${NUM_EXAMPLES}ex_${RING}ring_${ALGORITHM}_${RANDOM_ANGLES}random_${CONSTANT_ANGLES}constant${TAG}

. $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $CONSTANT_ANGLES $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO $PNM_NUM

XXX STOPPED HERE

### Training

### Analysis