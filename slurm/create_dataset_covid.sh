#!/bin/bash

#SBATCH -N 1            # Number of nodes
#SBATCH -J CT_NVAE_Create_Brain_Dataset      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m3562_g       # allocation account
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=gchen4@lbl.gov

export CT_NVAE_PATH=$1
export WORKING_DIR=$2
export EXAMPLE_NUMBER=$3
export SPARSE_NUM=$4
export RING_STRENGTH=$5
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH

mkdir -p $WORKING_DIR
cd $WORKING_DIR
export SLURM_NTASKS=4
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_images.py -n $EXAMPLE_NUMBER --dest images_covid --type covid
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_sinograms.py --dir images_covid

if [ "$RING_STRENGTH" -gt 0 ]; then
    python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_covid --dest dataset_covid_ring --train 0.7 --valid 0.2 --test 0.1 -n $EXAMPLE_NUMBER
    python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_covid_ring --sparse $SPARSE_NUM --random True --ring $RING_STRENGTH
else
    python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_covid --dest dataset_covid --train 0.7 --valid 0.2 --test 0.1 -n $EXAMPLE_NUMBER
    python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_covid --sparse $SPARSE_NUM --random True --ring 0
fi