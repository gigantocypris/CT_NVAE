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