#!/bin/bash

#SBATCH -N 1            # Number of nodes
#SBATCH -J CT_NVAE_Create_Dataset      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m3562_g       # allocation account
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:15:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=$1

module load python
export PYTHONPATH=$2:$PYTHONPATH
conda activate tomopy

export WORKING_DIR=$2
export CT_NVAE_PATH=$3
export SLURM_NTASKS=4
cd $WORKING_DIR

srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_images.py -n 64 --dest images_foam --type foam
srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_sinograms.py --dir images_foam
python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_foam --dest dataset_foam2 --train 0.7 --valid 0.2 --test 0.1 -n 64

python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_foam --dest dataset_foam_ring --train 0.7 --valid 0.2 --test 0.1 -n 64

python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_foam2 --sparse 20 --random True --ring 0

python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_foam_ring --sparse 20 --random True --ring 0.3