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

export CSV_PATH=/global/cfs/cdirs/m3562/users/hkim/brain_data/brain_merged_info.csv
export DCM_PATH=/global/cfs/cdirs/m3562/users/hkim/brain_data/raw/stage_2_train/
export THICKNESS=/global/cfs/cdirs/m3562/users/hkim/brain_data/instance_thickness.csv
export SLURM_NTASKS=4

mkdir -p $WORKING_DIR
cd $WORKING_DIR
export OUTPUT_PATH=$WORKING_DIR/images_brain
mkdir -p $OUTPUT_PATH
python $CT_NVAE_PATH/preprocessing/preprocess_brain.py $CSV_PATH $DCM_PATH $OUTPUT_PATH $THICKNESS $EXAMPLE_NUMBER

srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_sinograms.py --dir images_brain


if [ "$RING_STRENGTH" -gt 0 ]; then
    python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_brain --dest dataset_brain_ring --train 0.7 --valid 0.2 --test 0.1 -n $EXAMPLE_NUMBER
    python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_brain_ring --sparse $SPARSE_NUM --random True --ring $RING_STRENGTH
else
    python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_brain --dest dataset_brain --train 0.7 --valid 0.2 --test 0.1 -n $EXAMPLE_NUMBER
    python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_brain --sparse $SPARSE_NUM --random False --ring 0
fi