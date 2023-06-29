#!/bin/bash

#SBATCH -N 1            # Number of nodes
#SBATCH -J CT_NVAE_DATA      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -c 64
#SBATCH -o %j.out
#SBATCH -e %j.err

export NUM_TRAIN=$1 # number of images to create PER rank
export NUM_VAL=$2
export CT_NVAE_PATH=$3

export WORKING_DIR=$SCRATCH/output_CT_NVAE

cd $WORKING_DIR

echo "jobstart $(date)";pwd

srun -n 4 python $CT_NVAE_PATH/computed_tomography/create_foam_images.py -t $NUM_TRAIN -v $NUM_VAL

echo "jobend $(date)";pwd