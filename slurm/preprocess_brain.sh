#!/bin/bash

#SBATCH -N 1            # Number of nodes
#SBATCH -J preprocesss brain - temporary script      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m3562_g       # allocation account
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=hojune0630@lbl.gov

export CT_NVAE_PATH=$1
export WORKING_DIR=$2
export DATASET_ID=$3
export EXAMPLE_NUMBER=$4
export SPARSE_NUM=$5

export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
mkdir -p $WORKING_DIR
cd $WORKING_DIR

export SLURM_NTASKS=4
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_sinograms.py --dir $DATASET_ID
python $CT_NVAE_PATH/preprocessing/create_splits.py --src $DATASET_ID --dest dataset_$DATASET_ID --train 0.7 --valid 0.2 --test 0.1 -n $EXAMPLE_NUMBER
python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_$DATASET_ID --sparse $SPARSE_NUM --random False --ring 0

