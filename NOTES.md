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
Submitting job to train with foam_10ang_1000ex
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