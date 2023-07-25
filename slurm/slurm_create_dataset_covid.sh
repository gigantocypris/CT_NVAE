#!/bin/bash

#SBATCH -N 1            # Number of nodes
#SBATCH -J CT_NVAE_Create_Foam_Dataset      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m3562_g       # allocation account
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH -o %j.out
#SBATCH -e %j.err

export CT_NVAE_PATH=$1
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH

cd $WORKING_DIR
export SLURM_NTASKS=4
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_images.py -n $2 --dest images_covid --type covid
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_sinograms.py --dir images_covid

python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_covid --dest dataset_covid --train 0.7 --valid 0.2 --test 0.1 -n $2

python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_covid --dest dataset_covid_ring --train 0.7 --valid 0.2 --test 0.1 -n $2

python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_covid --sparse 20 --random True --ring 0

python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_covid_ring --sparse 20 --random True --ring 0.3