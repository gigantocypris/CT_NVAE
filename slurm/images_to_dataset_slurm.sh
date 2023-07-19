#!/bin/bash

#SBATCH -N 1            # Number of nodes
#SBATCH -J CT_NVAE_DATA      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH -o %j.out
#SBATCH -e %j.err

export NUM_IMG=$1 # number of images to process PER rank
export DATASET_TYPE=$2
export CT_NVAE_PATH=$3

export WORKING_DIR=$SCRATCH/output_CT_NVAE

cd $WORKING_DIR

echo "jobstart $(date)";pwd

srun -n 4 python $CT_NVAE_PATH/computed_tomography/images_to_dataset.py -n $NUM_IMG -d $DATASET_TYPE

echo "jobend $(date)";pwd