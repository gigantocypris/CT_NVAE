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
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export SLURM_NTASKS=4
cd $WORKING_DIR
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_sinograms.py --dir images_foam
```
Sinograms are created in the existing `images_foam` folder in the working directory `$WORKING_DIR`.

3. Split into training/test/validate
```
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_foam --dest dataset_foam2 --train 0.7 --valid 0.2 --test 0.1 -n 64
```
The split datasets are created in the `dataset_foam2` folder in the working directory `$WORKING_DIR`.

4. Create the dataset
```
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
cd $WORKING_DIR
python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_foam2 --sparse 20 --random True --ring 0
```
The dataset is created in the `dataset_foam2` folder in the working directory `$WORKING_DIR`.

5. Train the model.
First exit the interactive session, `conda deactivate`, and start a new one with the `CT_NVAE` environment:
```
module load python
conda activate CT_NVAE
salloc -N 1 --time=120 -C gpu -A m3562_g --qos=interactive --ntasks-per-gpu=1 --cpus-per-task=32
export WORKING_DIR=$SCRATCH/output_CT_NVAE
cd $WORKING_DIR
mkdir -p checkpts
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
```

Set environment variables for training:
```
export EXPR_ID=test_0000_foam2
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
python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam2 --batch_size 32 --epochs 100 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 --num_nf 0 --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --pnm 1e1
```

#### TODO: Convert to sbatch scripts
#### TODO: Run the tests in computed_tomography
#### TODO: Remove descriptions in the preprocessing scripts

### Full pass through the pipeline with the COVID images: